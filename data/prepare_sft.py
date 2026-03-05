"""Prepare SFT-only dataset: instruction pairs + companion loss mask.

Reads synthetic.jsonl (and any other ChatML-format JSONL files) and produces:
  data/sft/train.bin       — uint16 token IDs
  data/sft/train.mask.bin  — uint8 loss mask (1 = assistant token, 0 = prompt)
  data/sft/val.bin
  data/sft/val.mask.bin

Only ChatML records (those with a "messages" key) are included.
Documentation/plain-text records are skipped — they belong in pretraining.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from tokenizers import Tokenizer
from config import DataConfig

# Use centralized system prompt from config
SYSTEM_PROMPT = DataConfig.system_prompt


def format_chatml(messages: list[dict]) -> str:
    """Render a messages list to a ChatML string."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def build_mask(token_ids: list[int], im_start_id: int, im_end_id: int) -> list[int]:
    """Return a parallel mask: 1 for assistant-completion tokens, 0 otherwise.

    ChatML structure (system, user, assistant) means every third block
    (block_idx % 3 == 2) is the assistant turn.
    """
    mask = [0] * len(token_ids)
    block_idx = -1
    in_block = False
    skip_header = 0  # skip role token + newline after <|im_start|>

    for i, tok in enumerate(token_ids):
        if tok == im_start_id:
            block_idx += 1
            in_block = True
            skip_header = 2  # role token + newline
            continue
        if tok == im_end_id:
            in_block = False
            continue
        if in_block and skip_header > 0:
            skip_header -= 1
            continue
        if in_block and block_idx % 3 == 2:
            mask[i] = 1

    return mask


def load_jsonl_files(raw_dir: Path) -> list[dict]:
    """Load all JSONL files, keep only ChatML records (have 'messages' key)."""
    records = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Only keep ChatML instruction records
                if "messages" in obj and isinstance(obj["messages"], list):
                    records.append(obj)
    return records


def wrap_qa_as_chatml(obj: dict) -> dict | None:
    """Convert a Stack Overflow Q&A record to ChatML messages."""
    q = obj.get("question_title", "") + "\n" + obj.get("question_body", "")
    a = obj.get("answer_body", "")
    if not q.strip() or not a.strip():
        return None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q.strip()},
            {"role": "assistant", "content": a.strip()},
        ]
    }


def tokenize_and_mask(
    records: list[dict],
    tokenizer: Tokenizer,
    max_seq_len: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Tokenize ChatML records and build loss masks.

    Returns (all_tokens, all_masks) — one list per record.
    Long sequences are truncated to max_seq_len + 1 tokens.
    """
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")

    all_tokens = []
    all_masks = []
    skipped = 0

    for rec in records:
        msgs = rec["messages"]
        text = format_chatml(msgs)
        enc = tokenizer.encode(text)
        ids = enc.ids

        if len(ids) < 4:
            skipped += 1
            continue

        # Truncate to max_seq_len + 1 (we need one extra for the shifted target)
        ids = ids[: max_seq_len + 1]
        mask = build_mask(ids, im_start_id, im_end_id)

        # Skip records where mask is all zeros (no assistant tokens found)
        if sum(mask) == 0:
            skipped += 1
            continue

        all_tokens.append(ids)
        all_masks.append(mask)

    if skipped:
        print(f"  Skipped {skipped} records (too short or no assistant tokens)")

    return all_tokens, all_masks


def pack_and_save(
    token_seqs: list[list[int]],
    mask_seqs: list[list[int]],
    out_dir: Path,
    split: str,
    max_seq_len: int,
    pad_id: int = 2,
) -> int:
    """Pack sequences into fixed-length chunks and save .bin + .mask.bin files.

    Each chunk is exactly max_seq_len + 1 tokens. Sequences shorter than that
    are padded with pad_id (the <|pad|> token, ID 2). The mask is padded with 0.
    """
    chunk_len = max_seq_len + 1
    tokens_flat = []
    masks_flat = []

    for ids, mask in zip(token_seqs, mask_seqs):
        # Pad or truncate to chunk_len
        if len(ids) < chunk_len:
            ids = ids + [pad_id] * (chunk_len - len(ids))
            mask = mask + [0] * (chunk_len - len(mask))
        tokens_flat.append(ids[:chunk_len])
        masks_flat.append(mask[:chunk_len])

    tokens_arr = np.array(tokens_flat, dtype=np.uint16).reshape(-1)
    masks_arr = np.array(masks_flat, dtype=np.uint8).reshape(-1)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / f"{split}.bin"
    mask_path = out_dir / f"{split}.mask.bin"

    tokens_arr.tofile(tok_path)
    masks_arr.tofile(mask_path)

    n_chunks = len(tokens_flat)
    print(f"  Saved {tok_path}  ({len(tokens_arr):,} tokens, {n_chunks} samples)")
    print(f"  Saved {mask_path} ({int(masks_arr.sum()):,} assistant tokens, "
          f"{100*masks_arr.mean():.1f}% of total)")
    return n_chunks


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset with loss masks")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw JSONL directory")
    parser.add_argument("--output-dir", default="data/sft", help="Output directory")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--train-split", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)

    print(f"Loading ChatML records from {raw_dir} ...")
    records = load_jsonl_files(raw_dir)

    # Also wrap Q&A records that don't already have messages
    with_qa = 0
    for path in sorted(raw_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "question_title" in obj or "question_body" in obj:
                    wrapped = wrap_qa_as_chatml(obj)
                    if wrapped:
                        records.append(wrapped)
                        with_qa += 1

    print(f"  {len(records)} ChatML records found ({with_qa} from Q&A sources)")

    if not records:
        print("ERROR: No ChatML records found. Run generate_synthetic.py first.")
        sys.exit(1)

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(records)

    # Split
    n_train = int(len(records) * args.train_split)
    train_records = records[:n_train]
    val_records = records[n_train:]
    print(f"  Train: {len(train_records)} records, Val: {len(val_records)} records")

    # Load tokenizer
    print(f"\nTokenizing with {args.tokenizer} ...")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        pad_id = 2  # Default fallback
    print(f"  <|im_start|> id: {im_start_id},  <|im_end|> id: {im_end_id},  <|pad|> id: {pad_id}")

    train_tokens, train_masks = tokenize_and_mask(train_records, tokenizer, args.max_seq_len)
    val_tokens, val_masks = tokenize_and_mask(val_records, tokenizer, args.max_seq_len)

    print(f"\nSaving to {out_dir} ...")
    n_train_chunks = pack_and_save(train_tokens, train_masks, out_dir, "train", args.max_seq_len, pad_id)
    n_val_chunks = pack_and_save(val_tokens, val_masks, out_dir, "val", args.max_seq_len, pad_id)

    print(f"\n{'='*60}")
    print("SFT DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Train samples: {n_train_chunks:>8}")
    print(f"Val samples:   {n_val_chunks:>8}")
    print(f"Max seq len:   {args.max_seq_len:>8}")
    print(f"{'='*60}")
    print(f"\nRun SFT with:")
    print(f"  python -m training.train_sft \\")
    print(f"    --pretrain-checkpoint checkpoints/best.pt \\")
    print(f"    --data-dir {out_dir}")


if __name__ == "__main__":
    main()
