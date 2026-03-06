# model-nano

A small (~58M parameter) transformer LLM trained from scratch, specialized for Git/GitHub developer assistance. Runs on consumer GPUs (4GB VRAM).

## What It Does

- **Command generation** — Natural language to git commands ("undo my last commit" → `git reset --soft HEAD~1`)
- **Concept explanation** — Answer git questions with clear explanations
- **Command autocomplete** — Complete partial commands with suggestions

## Quick Start

### Install

```bash
# Using pipx (recommended for global CLI)
pipx install -e /path/to/model-nano

# Or with pip in a virtual environment
pip install -e ".[train]"
```

### Usage

```bash
# One-shot mode
git-nano "how do I squash the last 3 commits"

# Interactive REPL
git-nano

# Pipe mode
git status | git-nano explain

# As a git subcommand
git config --global alias.nano '!git-nano'
git nano "undo last commit"
```

### Action Prompt

After generating a command, an interactive menu appears:

| Key | Action |
|-----|--------|
| `Enter` | Execute the git command directly |
| `e` | Edit the command before running |
| `c` | Copy command to clipboard |
| `q` | Cancel (do nothing) |

Destructive commands (like `git reset --hard`) show a warning and require confirmation.

## Architecture

| Component | Value |
|-----------|-------|
| Parameters | ~58M |
| Layers | 32 |
| Hidden dim | 384 |
| Attention | GQA (8 heads, 4 KV heads) |
| FFN | SwiGLU |
| Position | RoPE |
| Context | 512 tokens |
| Vocab | 16,384 tokens (BPE) |

## Training

```bash
# 1. Collect and prepare data
python data/collect_docs.py
python data/generate_synthetic.py
python tokenizer/train_tokenizer.py
python data/prepare_dataset.py

# 2. Pre-train (next-token prediction)
python -m training.train_pretrain

# 3. Fine-tune (instruction following)
python -m training.train_sft --pretrain-checkpoint checkpoints/pretrain/best.pt

# 4. Evaluate
python -m eval.benchmark --model checkpoints/sft/best.pt
```

See `TRAINING_GUIDE.md` for detailed instructions.

## License

MIT
