"""Merge, deduplicate, tokenize, and split all data sources into binary files.

Inputs (JSONL files in data/raw/):
  - docs.jsonl       -- documentation text (type: book, manpage, tldr, github_docs)
  - stackoverflow.jsonl -- Q&A pairs (type: qa)
  - synthetic.jsonl  -- ChatML instruction pairs (messages array)

Processing pipeline:
  1. Load all sources
  2. Deduplicate (exact + normalized text matching)
  3. Format into ChatML for instruction data, raw text for pretraining data
  4. Tokenize using the project's BPE tokenizer (tokenizer/tokenizer.json)
  5. Split train/val (95/5)
  6. Save as memory-mapped .bin files (uint16 numpy arrays)
  7. Print statistics

Outputs:
  data/train.bin
  data/val.bin
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Add project root to path so we can import config
# ---------------------------------------------------------------------------
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DataConfig


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

def format_chatml(
    system: str,
    user: str,
    assistant: str,
) -> str:
    """Format a conversation turn into ChatML."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, returning a list of dicts.  Returns [] if missing."""
    if not os.path.isfile(path):
        print(f"  [skip] {path} not found")
        return []
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] {path}:{lineno}: {exc}")
    print(f"  Loaded {len(records)} records from {path}")
    return records


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip -- used for dedup."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Record -> text conversion
# ---------------------------------------------------------------------------

def record_to_text(record: dict, system_prompt: str) -> str:
    """Convert any record into a single training text string.

    - Synthetic data (has 'messages' key): already in ChatML format, extract fields.
    - Q&A data (has 'question_title'/'answer_body'): format as ChatML.
    - Documentation data (has 'text'): use raw text for pretraining.
    """
    # Synthetic ChatML data
    if "messages" in record:
        msgs = record["messages"]
        sys_msg = ""
        user_msg = ""
        asst_msg = ""
        for m in msgs:
            if m["role"] == "system":
                sys_msg = m["content"]
            elif m["role"] == "user":
                user_msg = m["content"]
            elif m["role"] == "assistant":
                asst_msg = m["content"]
        return format_chatml(sys_msg or system_prompt, user_msg, asst_msg)

    # Stack Overflow Q&A
    if "question_title" in record and "answer_body" in record:
        q = record.get("question_title", "")
        q_body = record.get("question_body", "")
        a = record.get("answer_body", "")
        user_content = q
        if q_body:
            user_content = f"{q}\n\n{q_body}"
        return format_chatml(system_prompt, user_content, a)

    # Documentation / raw text
    if "text" in record:
        return record["text"]

    # Fallback
    return json.dumps(record)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(records: list[dict], system_prompt: str) -> list[dict]:
    """Remove exact and normalized duplicates.  Returns deduplicated list."""
    seen_normalized: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        text = record_to_text(rec, system_prompt)
        norm = _normalize(text)
        if norm in seen_normalized:
            continue
        seen_normalized.add(norm)
        unique.append(rec)
    return unique


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    """Load the HuggingFace tokenizers Tokenizer from a JSON file."""
    from tokenizers import Tokenizer  # type: ignore[import-untyped]
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Train the tokenizer first (see tokenizer/ directory)."
        )
    tok = Tokenizer.from_file(tokenizer_path)
    return tok


def tokenize_texts(
    texts: list[str],
    tokenizer,
    max_seq_len: int,
) -> np.ndarray:
    """Tokenize a list of texts and pack into a flat uint16 array.

    Each text is tokenized independently. Sequences longer than max_seq_len
    are split into non-overlapping chunks.  We do NOT pad -- training will
    pack sequences densely.
    """
    all_ids: list[int] = []

    # Batch-encode for speed
    batch_size = 1024
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encodings = tokenizer.encode_batch(batch)
        for enc in encodings:
            ids = enc.ids
            # Split long sequences into chunks
            for chunk_start in range(0, len(ids), max_seq_len):
                chunk = ids[chunk_start : chunk_start + max_seq_len]
                all_ids.extend(chunk)

    return np.array(all_ids, dtype=np.uint16)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_bin(arr: np.ndarray, path: str) -> None:
    """Save a numpy array as a memory-mapped .bin file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Write raw bytes; can be loaded with np.memmap(path, dtype=np.uint16, mode='r')
    arr.tofile(path)
    size_mb = arr.nbytes / (1024 * 1024)
    print(f"  Saved {path}  ({len(arr):,} tokens, {size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(
    all_records: list[dict],
    deduped: list[dict],
    train_tokens: np.ndarray,
    val_tokens: np.ndarray,
) -> None:
    """Print a summary of the dataset."""
    from collections import Counter

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    # Source breakdown
    type_counter: Counter[str] = Counter()
    for rec in all_records:
        if "messages" in rec:
            type_counter["synthetic"] += 1
        elif "question_title" in rec:
            type_counter["stackoverflow"] += 1
        else:
            rtype = rec.get("type", "unknown")
            type_counter[rtype] += 1

    print(f"\nRaw records loaded:         {len(all_records):>8,}")
    print(f"After deduplication:        {len(deduped):>8,}")
    print(f"Removed duplicates:         {len(all_records) - len(deduped):>8,}")

    print("\nBreakdown by type:")
    for t, c in type_counter.most_common():
        print(f"  {t:20s} {c:>8,}")

    total_tokens = len(train_tokens) + len(val_tokens)
    print(f"\nTotal tokens:               {total_tokens:>12,}")
    print(f"  Train tokens:             {len(train_tokens):>12,}")
    print(f"  Val tokens:               {len(val_tokens):>12,}")
    print(f"  Train/Val ratio:          {len(train_tokens)/max(len(val_tokens),1):.1f}")
    print(f"\nTrain file size:            {train_tokens.nbytes/(1024*1024):>10.2f} MB")
    print(f"Val file size:              {val_tokens.nbytes/(1024*1024):>10.2f} MB")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge, deduplicate, tokenize, and split training data.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Directory containing raw JSONL files (default: from DataConfig)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write train.bin / val.bin (default: from DataConfig)",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="Path to tokenizer.json (default: from DataConfig)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length (default: from DataConfig)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=None,
        help="Fraction of data for training (default: from DataConfig)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from DataConfig)",
    )
    args = parser.parse_args()

    cfg = DataConfig()
    raw_dir = args.raw_dir or cfg.raw_dir
    output_dir = args.output_dir or cfg.processed_dir
    tokenizer_path = args.tokenizer_path or cfg.tokenizer_path
    max_seq_len = args.max_seq_len or cfg.max_seq_len
    train_split = args.train_split or cfg.train_split
    seed = args.seed if args.seed is not None else cfg.seed
    system_prompt = cfg.system_prompt

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Load all sources
    # ------------------------------------------------------------------
    print("Loading data sources ...")
    docs = load_jsonl(os.path.join(raw_dir, "docs.jsonl"))
    stackoverflow = load_jsonl(os.path.join(raw_dir, "stackoverflow.jsonl"))
    synthetic = load_jsonl(os.path.join(raw_dir, "synthetic.jsonl"))

    all_records = docs + stackoverflow + synthetic
    if not all_records:
        print("No records found. Nothing to do.")
        return

    # ------------------------------------------------------------------
    # 2. Deduplicate
    # ------------------------------------------------------------------
    print("\nDeduplicating ...")
    deduped = deduplicate(all_records, system_prompt)
    print(f"  {len(all_records)} -> {len(deduped)} records ({len(all_records) - len(deduped)} duplicates removed)")

    # ------------------------------------------------------------------
    # 3. Convert to text
    # ------------------------------------------------------------------
    print("\nConverting records to text ...")
    texts: list[str] = []
    for rec in deduped:
        text = record_to_text(rec, system_prompt)
        text = text.strip()
        if text:
            texts.append(text)
    print(f"  {len(texts)} non-empty text sequences")

    # ------------------------------------------------------------------
    # 4. Tokenize
    # ------------------------------------------------------------------
    print(f"\nTokenizing with {tokenizer_path} ...")
    tokenizer = load_tokenizer(tokenizer_path)
    all_token_ids = tokenize_texts(texts, tokenizer, max_seq_len)
    print(f"  Total token count: {len(all_token_ids):,}")

    # ------------------------------------------------------------------
    # 5. Shuffle & split
    # ------------------------------------------------------------------
    print("\nShuffling and splitting ...")
    # Shuffle at the document level by shuffling texts, then re-tokenizing.
    # For efficiency, we shuffle the token array in max_seq_len-sized blocks.
    n_tokens = len(all_token_ids)
    # Trim to exact multiple of max_seq_len for clean chunks
    n_chunks = n_tokens // max_seq_len
    trimmed = all_token_ids[: n_chunks * max_seq_len]
    chunks = trimmed.reshape(n_chunks, max_seq_len)
    perm = rng.permutation(n_chunks)
    chunks = chunks[perm]

    split_idx = int(n_chunks * train_split)
    train_chunks = chunks[:split_idx]
    val_chunks = chunks[split_idx:]

    train_tokens = train_chunks.reshape(-1)
    val_tokens = val_chunks.reshape(-1)

    # Also include the remainder tokens (that didn't fill a full chunk) in train
    remainder = all_token_ids[n_chunks * max_seq_len :]
    if len(remainder) > 0:
        train_tokens = np.concatenate([train_tokens, remainder])

    print(f"  Train: {len(train_tokens):,} tokens  ({split_idx} chunks)")
    print(f"  Val:   {len(val_tokens):,} tokens  ({n_chunks - split_idx} chunks)")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print("\nSaving binary files ...")
    os.makedirs(output_dir, exist_ok=True)
    save_bin(train_tokens, os.path.join(output_dir, "train.bin"))
    save_bin(val_tokens, os.path.join(output_dir, "val.bin"))

    # ------------------------------------------------------------------
    # 7. Statistics
    # ------------------------------------------------------------------
    print_statistics(all_records, deduped, train_tokens, val_tokens)


if __name__ == "__main__":
    main()
