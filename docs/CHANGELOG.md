# Changelog

All notable changes to model-nano are documented in this file.

---

## [0.1.0] - 2026-03-02

### Initial Implementation

The complete project was built from scratch in a single session. Everything below represents the v0.1.0 baseline.

### Added

#### Model Architecture (`model/`)
- NanoGPT transformer with ~58M parameters (32 layers, 384 hidden dim)
- Grouped-Query Attention (8 query heads, 4 KV heads) with KV-cache support
- RMSNorm (pre-norm), RoPE positional encoding, SwiGLU FFN
- Weight tying between token embeddings and output projection
- Scaled residual initialization per GPT-2 paper

#### Data Pipeline (`data/`)
- `collect_docs.py` — Clone and parse Pro Git book, git man pages, tldr-pages, GitHub Docs
- `collect_stackoverflow.py` — Parse Stack Overflow XML dump, filter git-tagged posts (score >= 5)
- `generate_synthetic.py` — Three synthetic data strategies:
  - Seed expansion: 50 seed commands with 10+ natural language variations each
  - Error scenarios: 15 common git errors with diagnosis and fix
  - Flag combinatorics: 100+ flag combinations across 15 git commands
- `prepare_dataset.py` — Merge, deduplicate, tokenize, split into train/val .bin files

#### Tokenizer (`tokenizer/`)
- Byte-level BPE tokenizer using HuggingFace `tokenizers` library
- 16,384 vocab size target
- Special tokens: `<|im_start|>`, `<|im_end|>`, `<|pad|>`
- Fallback seed corpus for training without external data

#### Training Pipeline (`training/`)
- `train_pretrain.py` — Phase 1 next-token prediction with:
  - AdamW optimizer (beta1=0.9, beta2=0.95, weight_decay=0.1)
  - Cosine LR schedule with linear warmup (2% of steps)
  - BF16 mixed precision, gradient checkpointing, gradient accumulation
  - Periodic evaluation, best-model checkpointing, optional wandb logging
- `train_sft.py` — Phase 2 supervised fine-tuning with:
  - Loss masking (train on assistant completions only)
  - Lower learning rate (2e-5)
  - Loads from Phase 1 checkpoint
- `dataset.py` — Memory-mapped PretrainDataset and SFTDataset
- `utils.py` — Cosine scheduler, checkpoint save/load, TrainingLogger

#### Inference Engine (`inference/`)
- `engine.py` — KV-cache inference engine for fast autoregressive generation
- `generate.py` — Top-k, top-p (nucleus), temperature sampling strategies
- `export_onnx.py` — ONNX export with INT8 dynamic quantization

#### CLI Tool (`cli/`)
- `__main__.py` — Click-based entry point with natural subcommand routing
- `oneshot.py` — Suggest-then-confirm UX (Execute / Edit / Copy / Cancel)
- `interactive.py` — REPL chat mode with conversation history
- `context.py` — Git repo state detection, destructive command warnings

#### Evaluation (`eval/`)
- `benchmark.py` — Full benchmark runner with rich table output
- `metrics.py` — Exact match, command equivalence (handles flag aliases, command synonyms), response quality scoring
- `test_cases.json` — 50 hand-curated test cases across 8 categories

#### Configuration & Packaging
- `config.py` — Dataclass configs for model, training, SFT, and data
- `pyproject.toml` — Package metadata, entry point (`git-nano`), optional dependency groups
- `requirements.txt` — All Python dependencies
- `.gitignore` — Excludes data, checkpoints, artifacts

### Verified
- Model forward pass produces expected loss (~9.7 = ln(16384))
- KV-cache autoregressive decode works correctly
- Tokenizer round-trips all test strings perfectly
- Training loop runs and loss decreases
- Synthetic data generates 1000+ pairs from hardcoded seeds
- Dataset pipeline: merge -> dedup -> tokenize -> split
- All sampling strategies (greedy, top-k, top-p) produce valid outputs
- Eval metrics correctly identify equivalent git commands
- Destructive command detection catches force-push, reset --hard, etc.

---

## [0.1.1] - 2026-03-03

### First Real Training Run — Phase 1 Pre-training Complete

#### Dataset (actual numbers from run)

| Source | Records | Tokens |
|--------|---------|--------|
| Pro Git book | 106 docs | — |
| tldr-pages | 202 docs | — |
| GitHub Docs | 1,412 docs | — |
| Synthetic | 5,000 pairs | — |
| **After dedup** | **2,634 records** | **1,684,795 tokens** |

- Git man pages: **0 documents** parsed (bug — AsciiDoc parser missed them, needs fix)
- Stack Overflow: not collected (no data dump available)
- Tokenizer: full 16,384 vocab achieved from combined corpus
- Train split: 1,600,315 tokens (3,125 chunks), Val split: 84,480 tokens (165 chunks)

#### Phase 1 Pre-training Results

| Metric | Value |
|--------|-------|
| Total steps | 960 |
| Epochs | 20 |
| Total tokens processed | 31,907,840 (~32M) |
| Training time | ~6,060s (~1h 41m) |
| GPU memory peak | 795 MB / 4,096 MB |
| Speed | ~5,000–5,300 tok/s |
| Initial train loss | 8.96 (step 10) |
| Final train loss | 0.62 (step 960) |
| Best val loss | 3.19 (step 500) — saved as `checkpoints/best.pt` |
| Final checkpoint | `checkpoints/final.pt` |

#### Analysis

**What went well:**
- Loss decreased smoothly from 8.96 → 0.62 with no NaN/explosion
- GPU memory stayed at 795 MB — well within the 4GB budget
- Stable throughput (~5,200 tok/s consistent throughout)
- Gradient norm well-controlled (0.3–0.9 range after warmup)
- BF16 + gradient checkpointing working correctly

**Overfitting — expected at this data scale:**
- Train loss (0.62) vs val loss (3.19 at step 500) shows significant overfitting
- Root cause: only 1.6M training tokens with a 58M parameter model. The model has enough capacity to near-memorize the training set.
- The ratio is ~27 parameters per training token — typical guidance is 10–20 tokens per parameter (Chinchilla). We need ~600M tokens for compute-optimal training at this size, or ~6B for "overtrained" inference-optimal training.
- **Best checkpoint is `checkpoints/best.pt` (val_loss 3.19, step 500)**, not `final.pt`

**Next step: run SFT on best.pt, then massively expand the dataset.**

---

## [0.1.2] - 2026-03-03

### Pipeline Automation + Data & Bug Fixes

#### New Files
- **`train.sh`** — Full 8-step pipeline script with resume (`--from stepN`), skip flags, elapsed timing, and colored output
- **`data/prepare_sft.py`** — NEW: creates `data/sft/train.bin` + `data/sft/train.mask.bin` with correct ChatML loss masks (only assistant tokens get loss)

#### Bug Fixes

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| SFT loss = 0.0000 | `IM_START_ID=16381` wrong (actual id=0), causing runtime mask to never activate | Fixed to `IM_START_ID=0`, `IM_END_ID=1` in `training/dataset.py` |
| SFT no improvement (mixed data) | `data/train.bin` mixed docs + instructions; most 512-token chunks contained no ChatML structure | Created `prepare_sft.py` to generate SFT-only data in `data/sft/` |
| GitHub Docs AUTOTITLE spam | `_strip_markdown` converted `[AUTOTITLE](/path)` → `"AUTOTITLE"` in training text | Added explicit AUTOTITLE removal + Liquid template stripping in `collect_docs.py` |
| Git man pages: 0 documents | git/git repo migrated from `.txt` to `.adoc` extensions in 2024 | `collect_git_manpages()` now globs both `*.txt` and `*.adoc` |
| `--count` ambiguous argument | `generate_synthetic.py` had `--count-seed/errors/flags` so `--count` was ambiguous | Added explicit `--count N` flag splitting 40/20/40 across strategies |
| DataLoader deadlock (num_workers=4) | Fork-based multiprocessing conflicts after prior CUDA initialization | Changed `create_dataloader` default to `num_workers=0`, `persistent_workers=False` |
| Python output buffering via `tee` | `exec > >(tee -a log)` + Python's default 8KB buffer = no visible output during training | Added `python -u` (unbuffered) to all training commands in `train.sh` |
| `sample_top_p scatter_` dimension error | `logits.scatter_(0, ...)` wrong for 2D tensors | Changed to `scatter_(-1, ...)` in `inference/generate.py` |
| CLI default model path not found | Hardcoded `checkpoints/model.pt` didn't exist | Auto-detect from ordered candidates: `sft/best.pt → sft/final.pt → best.pt → final.pt` |

#### Second Training Run (in progress as of 2026-03-03 08:31)

Pipeline: `./train.sh --from step7 --synthetic-count 20000`

Dataset (rebuilt with fixes):
- 1,958 docs (Pro Git + git man pages with .adoc fix + tldr + GitHub Docs cleaned)
- 20,000 synthetic pairs (4x larger than first run)
- Pretrain: ~1.97M tokens in `data/train.bin`
- SFT: ~9.7M tokens in `data/sft/train.bin` with companion `.mask.bin`

Training status (estimated):
- Phase 1: 1,180 total steps, ~2.1 hours, training at 100% GPU / 1850MB VRAM
- Expected completion: ~10:38 AM
- Phase 2 SFT to follow after Phase 1 checkpoint

---

## Known Issues

1. **Tokenizer vocab size** — When trained only on synthetic data (no external docs), the vocab is ~1,500 tokens instead of the target 16,384. This is expected; the full vocab is reached when training on the complete corpus (Pro Git + man pages + SO + synthetic).

2. **Parameter count** — The model has ~58M parameters instead of the original ~45M target. The difference is because SwiGLU uses 3 weight matrices per FFN layer (gate, value, down) instead of 2. The architecture dimensions (32 layers, 384 hidden, 1024 FFN) are as specified. The model still fits comfortably in 4GB VRAM.

3. **No pre-trained weights** — The model needs to be trained before the CLI is useful. Without weights, the CLI will output random tokens.

---

## Needs Improvement

See the roadmap in [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical improvements.

### High Priority
- [x] Collect real-world data corpus (Pro Git, tldr, GitHub Docs) ✓
- [x] Train tokenizer on full corpus — 16,384 vocab achieved ✓
- [x] Complete Phase 1 pre-training run ✓ (val_loss 3.19)
- [x] **Fix git man pages parser** — now parses both `.txt` and `.adoc` ✓
- [x] **Fix GitHub Docs AUTOTITLE/Liquid template noise** ✓
- [x] **Fix SFT loss masking** — correct ChatML token IDs + `prepare_sft.py` ✓
- [x] **Expand synthetic data** — 5K → 20K pairs ✓
- [x] **Automated pipeline** — `train.sh` runs all 8 steps with resume support ✓
- [ ] **Collect Stack Overflow data** — biggest quality signal missing (~30K-50K pairs)
- [ ] Complete Phase 2 SFT run (in progress)
- [ ] Run benchmark and establish baseline accuracy

### Medium Priority
- [ ] Add more synthetic data variety (multi-step workflows, edge cases)
- [ ] Add Evol-Instruct synthetic data generation (progressively complex queries)
- [ ] Implement git alias integration (`git nano`)
- [ ] Add shell completions (bash, zsh, fish)
- [ ] Add `--dry-run` flag to CLI for command preview without execution
- [ ] Expand test cases from 50 to 300

### Low Priority
- [ ] GGUF export for llama.cpp compatibility
- [ ] DPO/RLHF alignment pass
- [ ] Streaming token output in CLI
- [ ] Plugin system for extending command support
- [ ] Model distillation from a larger teacher model
- [ ] Web playground for browser-based demo
