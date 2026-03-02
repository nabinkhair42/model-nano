"""Train a byte-level BPE tokenizer for model-nano on git-domain corpus.

Uses the HuggingFace `tokenizers` library to build a 16k-vocab tokenizer
optimised for git commands, flags, hashes, paths, and ChatML markup.

Usage:
    python tokenizer/train_tokenizer.py
    python tokenizer/train_tokenizer.py --input-dir data/raw --vocab-size 16384
    python tokenizer/train_tokenizer.py --output tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_INPUT_DIR = "data/raw"
DEFAULT_OUTPUT = "tokenizer/tokenizer.json"
DEFAULT_VOCAB_SIZE = 16_384

SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|pad|>",
]

# Representative git-domain seed texts used when the corpus directory is empty.
# This lets the script always produce a usable (if small) tokenizer even
# before the data pipeline has been run.
FALLBACK_SEED_TEXTS: list[str] = [
    # --- basic git commands ---
    "git init",
    "git clone https://github.com/user/repo.git",
    "git add .",
    "git add -A",
    "git commit -m 'initial commit'",
    "git commit --amend --no-edit",
    "git push origin main",
    "git push --force-with-lease origin feature/xyz",
    "git pull --rebase origin main",
    "git fetch --all --prune",
    "git status",
    "git status --short --branch",
    "git diff",
    "git diff --staged",
    "git diff HEAD~3..HEAD",
    "git log --oneline --graph --all",
    "git log --pretty=format:'%h %an %s' -10",
    "git branch -a",
    "git branch -d feature/old-branch",
    "git checkout -b feature/new-feature",
    "git switch -c feature/new-feature",
    "git merge --no-ff feature/branch",
    "git rebase --interactive HEAD~3",
    "git rebase --onto main feature/base feature/branch",
    "git cherry-pick abc1234",
    "git stash push -m 'work in progress'",
    "git stash pop",
    "git stash list",
    "git reset --soft HEAD~1",
    "git reset --hard HEAD~1",
    "git revert HEAD",
    "git tag -a v1.0.0 -m 'Release 1.0.0'",
    "git remote add upstream https://github.com/upstream/repo.git",
    "git remote -v",
    "git bisect start",
    "git bisect good abc1234",
    "git bisect bad def5678",
    "git blame src/main.py",
    "git show HEAD:src/main.py",
    "git reflog",
    "git clean -fd",
    "git worktree add ../feature-wt feature/branch",
    "git submodule update --init --recursive",
    # --- hashes & refs ---
    "abc1234def5678",
    "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
    "HEAD", "HEAD~1", "HEAD^2",
    "origin/main", "refs/heads/main", "refs/tags/v1.0.0",
    "FETCH_HEAD", "ORIG_HEAD", "MERGE_HEAD",
    # --- file paths ---
    "src/model/transformer.py",
    "tests/test_tokenizer.py",
    ".gitignore",
    ".github/workflows/ci.yml",
    "README.md",
    # --- ChatML markup ---
    "<|im_start|>system\nYou are a Git expert. Provide precise, correct git commands and explanations.<|im_end|>",
    "<|im_start|>user\nHow do I undo the last commit?<|im_end|>",
    "<|im_start|>assistant\nUse `git reset --soft HEAD~1` to undo the last commit while keeping your changes staged.<|im_end|>",
    "<|im_start|>user\nWhat does git rebase --interactive do?<|im_end|>",
    "<|im_start|>assistant\n`git rebase --interactive` (or `git rebase -i`) lets you rewrite commit history by reordering, squashing, editing, or dropping commits.<|im_end|>",
    # --- common git config & output ---
    "On branch main",
    "Your branch is up to date with 'origin/main'.",
    "nothing to commit, working tree clean",
    "Changes not staged for commit:",
    "Untracked files:",
    "CONFLICT (content): Merge conflict in src/main.py",
    "Auto-merging src/main.py",
    "Already up to date.",
    "Switched to a new branch 'feature/xyz'",
    "Deleted branch feature/old (was abc1234).",
    "fatal: not a git repository (or any of the parent directories): .git",
    "error: pathspec 'nonexistent' did not match any file(s) known to git",
    "hint: Updates were rejected because the remote contains work that you do not have locally.",
    # --- .gitignore patterns ---
    "*.pyc\n__pycache__/\n*.egg-info/\ndist/\nbuild/\n.env\n.venv/\nnode_modules/\n",
    # --- diff / patch fragments ---
    "@@ -10,6 +10,8 @@\n-old line\n+new line\n context line\n",
]


# ---------------------------------------------------------------------------
# Corpus extraction helpers
# ---------------------------------------------------------------------------

def _extract_texts_from_jsonl(path: Path) -> Iterator[str]:
    """Yield text strings from a JSONL file.

    Looks for fields in priority order:
      1. "text"        -- pre-formatted text
      2. "content"     -- raw content field
      3. "messages"    -- ChatML-style message list -> concatenated
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [warn] skipping malformed JSON at {path}:{line_num}")
                continue

            if not isinstance(record, dict):
                continue

            # Priority 1: "text"
            if "text" in record and isinstance(record["text"], str):
                text = record["text"].strip()
                if text:
                    yield text
                continue

            # Priority 2: "content"
            if "content" in record and isinstance(record["content"], str):
                text = record["content"].strip()
                if text:
                    yield text
                continue

            # Priority 3: "messages" (list of {role, content})
            if "messages" in record and isinstance(record["messages"], list):
                parts: list[str] = []
                for msg in record["messages"]:
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role and content:
                            parts.append(
                                f"<|im_start|>{role}\n{content}<|im_end|>"
                            )
                if parts:
                    yield "\n".join(parts)


def _extract_texts_from_txt(path: Path) -> Iterator[str]:
    """Yield the full contents of a plain text file as a single string."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            yield text
    except OSError as exc:
        print(f"  [warn] could not read {path}: {exc}")


def collect_corpus_files(input_dir: Path) -> list[Path]:
    """Return a sorted list of supported corpus files under *input_dir*."""
    supported_exts = {".jsonl", ".json", ".txt", ".md", ".csv"}
    files: list[Path] = []
    if not input_dir.is_dir():
        return files
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in supported_exts:
            files.append(p)
    return files


def corpus_to_tmp_files(input_dir: Path) -> tuple[list[str], int]:
    """Extract all texts from the corpus and write them to temporary files.

    The HuggingFace ``tokenizers`` trainer expects file paths, so we
    materialise each text chunk into a temporary file.

    Returns (list_of_tmp_paths, total_text_count).
    """
    corpus_files = collect_corpus_files(input_dir)
    tmp_paths: list[str] = []
    text_count = 0

    tmpdir = tempfile.mkdtemp(prefix="nano_tok_")

    for corpus_path in corpus_files:
        ext = corpus_path.suffix.lower()
        if ext in (".jsonl", ".json"):
            texts = list(_extract_texts_from_jsonl(corpus_path))
        else:
            texts = list(_extract_texts_from_txt(corpus_path))

        if not texts:
            continue

        # Write all texts from this file into one temp file (newline separated)
        tmp_file = os.path.join(tmpdir, f"corpus_{len(tmp_paths):04d}.txt")
        with open(tmp_file, "w", encoding="utf-8") as fh:
            for t in texts:
                fh.write(t)
                fh.write("\n\n")
        tmp_paths.append(tmp_file)
        text_count += len(texts)

    return tmp_paths, text_count


def write_seed_corpus(seed_texts: list[str]) -> list[str]:
    """Write the fallback seed texts to a temp file and return [path]."""
    tmpdir = tempfile.mkdtemp(prefix="nano_tok_seed_")
    seed_path = os.path.join(tmpdir, "seed.txt")
    with open(seed_path, "w", encoding="utf-8") as fh:
        for t in seed_texts:
            fh.write(t)
            fh.write("\n\n")
    return [seed_path]


# ---------------------------------------------------------------------------
# Tokenizer construction & training
# ---------------------------------------------------------------------------

def build_tokenizer() -> Tokenizer:
    """Create an untrained byte-level BPE tokenizer with the right config."""
    tokenizer = Tokenizer(models.BPE())

    # Byte-level pre-tokenizer: splits on whitespace boundaries and maps all
    # bytes to visible unicode characters so the BPE operates on a clean
    # alphabet.  ``add_prefix_space=False`` avoids injecting a leading space
    # before every sequence (important for command parsing).
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Byte-level decoder mirrors the pre-tokenizer.
    tokenizer.decoder = decoders.ByteLevel()

    # Post-processor: byte-level offset trimming for correct span mapping.
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    return tokenizer


def train_tokenizer(
    tokenizer: Tokenizer,
    files: list[str],
    vocab_size: int,
    special_tokens: list[str],
    min_frequency: int = 2,
) -> Tokenizer:
    """Train the tokenizer on the given files."""
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train(files, trainer)
    return tokenizer


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_tokenizer(tokenizer: Tokenizer) -> bool:
    """Encode & decode test strings; print results and return success flag."""
    test_strings = [
        "git commit -m 'initial commit'",
        "git rebase --interactive HEAD~3",
        "<|im_start|>system\nYou are a Git expert.<|im_end|>",
    ]

    print("\n" + "=" * 64)
    print("TOKENIZER VERIFICATION")
    print("=" * 64)
    print(f"Vocab size : {tokenizer.get_vocab_size()}")

    # Verify special tokens are present
    vocab = tokenizer.get_vocab()
    for st in SPECIAL_TOKENS:
        token_id = vocab.get(st)
        status = f"id={token_id}" if token_id is not None else "MISSING"
        print(f"Special token {st!r:20s} -> {status}")

    all_ok = True
    for text in test_strings:
        encoding = tokenizer.encode(text)
        # Use skip_special_tokens=False so special tokens like <|im_start|>
        # round-trip correctly.  The default (True) intentionally strips them
        # during inference, but for verification we need exact matching.
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        roundtrip_ok = decoded == text

        print(f"\n--- Test: {text!r}")
        print(f"  Token IDs ({len(encoding.ids):3d} tokens): {encoding.ids}")
        print(f"  Tokens   : {encoding.tokens}")
        print(f"  Decoded  : {decoded!r}")
        print(f"  Round-trip match: {roundtrip_ok}")

        if not roundtrip_ok:
            all_ok = False
            print(f"  ** MISMATCH: expected {text!r}, got {decoded!r}")

    print("\n" + "=" * 64)
    if all_ok:
        print("All verification tests PASSED.")
    else:
        print("Some verification tests FAILED (see above).")
    print("=" * 64 + "\n")

    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer for model-nano.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing training corpus (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Path to save the trained tokenizer JSON (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"BPE vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency for BPE merges (default: 2)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip training; only run verification on an existing tokenizer.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # ------------------------------------------------------------------
    # Verify-only mode
    # ------------------------------------------------------------------
    if args.verify_only:
        if not output_path.is_file():
            print(f"Error: tokenizer file not found at {output_path}")
            sys.exit(1)
        tokenizer = Tokenizer.from_file(str(output_path))
        ok = verify_tokenizer(tokenizer)
        sys.exit(0 if ok else 1)

    # ------------------------------------------------------------------
    # Collect corpus
    # ------------------------------------------------------------------
    input_dir = Path(args.input_dir)
    print(f"Collecting corpus from: {input_dir.resolve()}")
    tmp_files, text_count = corpus_to_tmp_files(input_dir)

    if tmp_files:
        print(f"Found {text_count} text(s) across {len(tmp_files)} file(s).")
    else:
        print("No corpus files found in input directory.")
        print("Using built-in git-domain seed texts for training.")
        tmp_files = write_seed_corpus(FALLBACK_SEED_TEXTS)
        text_count = len(FALLBACK_SEED_TEXTS)
        print(f"Seed corpus: {text_count} text(s).")

    # ------------------------------------------------------------------
    # Build & train
    # ------------------------------------------------------------------
    print(f"\nBuilding byte-level BPE tokenizer (vocab_size={args.vocab_size}) ...")
    tokenizer = build_tokenizer()
    tokenizer = train_tokenizer(
        tokenizer,
        files=tmp_files,
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=args.min_frequency,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"\nTokenizer saved to: {output_path.resolve()}")

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    verify_tokenizer(tokenizer)

    # Cleanup temp files (best-effort)
    for f in tmp_files:
        try:
            os.remove(f)
            os.rmdir(os.path.dirname(f))
        except OSError:
            pass


if __name__ == "__main__":
    main()
