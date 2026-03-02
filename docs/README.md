# model-nano

A small (~58M parameter) transformer LLM trained from scratch, specialized for Git/GitHub developer assistance. Designed to run on consumer GPUs (RTX 3050, 4GB VRAM) and distributed as a standalone CLI tool.

## What It Does

model-nano handles three core tasks:

1. **Command generation** — Natural language to git commands ("undo my last commit" -> `git reset --soft HEAD~1`)
2. **Concept explanation** — Answer git questions with clear explanations
3. **Command autocomplete** — Complete partial commands with ranked suggestions

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Parameters | ~58M total (6.3M embedding, 51.9M transformer) | Small enough for 4GB VRAM, large enough to be useful |
| Layers | 32 | Deep networks outperform wide ones at this scale (HuggingFace Dhara-70M study) |
| Hidden dim | 384 | Balanced with depth to avoid the <512 hidden "dead zone" |
| FFN | SwiGLU, 1024 intermediate | Higher expressivity via gating, standard in LLaMA-family |
| Attention | GQA (8 heads, 4 KV heads) | Saves memory with negligible quality loss |
| Norm | RMSNorm (pre-norm) | 10-50% more efficient than LayerNorm |
| Position | RoPE (theta=10000) | Better length generalization than learned embeddings |
| Vocab | 16,384 tokens (BPE) | Keeps embedding params at 14% of total |
| Context | 512 tokens | Sufficient for git commands and short explanations |

## Project Structure

```
model-nano/
├── config.py                     # All hyperparameters (model, training, data)
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Package config (pip install -e .)
├── .gitignore
│
├── model/                        # Transformer architecture
│   ├── components.py             #   RMSNorm, RoPE, SwiGLU FFN
│   ├── attention.py              #   Grouped-query attention + KV cache
│   └── transformer.py            #   Full NanoGPT model
│
├── data/                         # Data pipeline
│   ├── collect_docs.py           #   Clone & parse Pro Git, man pages, tldr, GH docs
│   ├── collect_stackoverflow.py  #   Parse SO data dump for git Q&A
│   ├── generate_synthetic.py     #   Synthetic data (seed expansion, errors, flags)
│   └── prepare_dataset.py        #   Merge, dedupe, tokenize, split to .bin
│
├── tokenizer/
│   ├── train_tokenizer.py        #   Train byte-level BPE tokenizer
│   └── tokenizer.json            #   Saved tokenizer (after training)
│
├── training/                     # Training pipeline
│   ├── dataset.py                #   PyTorch Dataset/DataLoader
│   ├── utils.py                  #   LR scheduler, checkpointing, logging
│   ├── train_pretrain.py         #   Phase 1: next-token prediction on docs
│   └── train_sft.py              #   Phase 2: supervised fine-tuning
│
├── inference/                    # Inference engine
│   ├── engine.py                 #   KV-cache inference engine
│   ├── generate.py               #   Sampling strategies (top-k, top-p, greedy)
│   └── export_onnx.py            #   ONNX export + INT8 quantization
│
├── eval/                         # Evaluation
│   ├── benchmark.py              #   Run git command benchmark
│   ├── metrics.py                #   Exact match, command equivalence, quality
│   └── test_cases.json           #   50 hand-curated test cases
│
├── cli/                          # CLI tool
│   ├── __main__.py               #   Entry point (click-based)
│   ├── oneshot.py                #   One-shot query with suggest-then-confirm UX
│   ├── interactive.py            #   REPL chat mode
│   └── context.py                #   Git repo context detection
│
├── docs/                         # Documentation
│   ├── README.md                 #   This file
│   ├── CHANGELOG.md              #   Version history and changes
│   ├── ARCHITECTURE.md           #   Technical architecture deep-dive
│   ├── TRAINING_GUIDE.md         #   How to train the model
│   └── DATA_GUIDE.md             #   Data collection and preparation
│
└── checkpoints/                  # Saved model weights (gitignored)
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.1+ with CUDA support (for GPU training)
- ~4GB VRAM (RTX 3050 or equivalent)

### Install

```bash
cd model-nano
pip install -e ".[train]"
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### Full Pipeline

```bash
# 1. Collect real documentation
python data/collect_docs.py

# 2. Generate synthetic training data
python data/generate_synthetic.py

# 3. Train the tokenizer
python tokenizer/train_tokenizer.py

# 4. Prepare the dataset (merge, tokenize, split)
python data/prepare_dataset.py

# 5. Phase 1: Pre-train on documentation (next-token prediction)
python -m training.train_pretrain

# 6. Phase 2: Fine-tune on instruction pairs
python -m training.train_sft --pretrain-checkpoint checkpoints/pretrain/best.pt

# 7. Run the CLI
git-nano "undo my last commit"
```

### CLI Usage

```bash
# One-shot mode
git-nano "how do I squash the last 3 commits"

# Interactive REPL
git-nano

# Pipe mode
git status | git-nano explain

# As a git subcommand (add alias)
git config --global alias.nano '!git-nano'
git nano "undo last commit"
```

### Evaluation

```bash
python -m eval.benchmark --model checkpoints/sft/best.pt --verbose
```

## Training Details

### Two-Phase Training

**Phase 1 — Pre-training**: Next-token prediction on git documentation (Pro Git book, man pages, tutorials). Teaches the model git vocabulary, syntax, and language patterns.

**Phase 2 — Instruction Fine-tuning (SFT)**: Train on instruction-response pairs in ChatML format. Only trains on assistant completions (system/user tokens are masked). Uses lower learning rate (2e-5).

### Data Sources

| Source | Type | Est. Pairs |
|--------|------|-----------|
| Pro Git book | Explanations, workflows | ~5,000-10,000 |
| Git man pages | Command reference | ~10,000-20,000 |
| tldr-pages | Structured examples | ~500-800 |
| GitHub Docs | API, workflows | ~5,000-10,000 |
| Stack Overflow | Community Q&A | ~30,000-50,000 |
| Synthetic (seed expansion) | Command variations | ~20,000-50,000 |
| Synthetic (error scenarios) | Error diagnosis | ~10,000-20,000 |
| Synthetic (flag combos) | Flag reference | ~20,000-50,000 |

**Target: ~100,000-200,000 high-quality pairs**

### Memory Budget (4GB VRAM)

| Component | Size |
|-----------|------|
| Model weights (BF16) | ~116 MB |
| Optimizer states (FP32) | ~464 MB |
| Gradients (BF16) | ~116 MB |
| Activations (with checkpointing) | ~500-1000 MB |
| Framework overhead | ~200 MB |
| **Total** | **~1.4-1.9 GB** |

## License

MIT
