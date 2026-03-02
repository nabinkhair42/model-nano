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
- [ ] **Fix git man pages parser** — 0/168 man pages parsed (AsciiDoc `.txt` detection broken)
- [ ] **Collect Stack Overflow data** — biggest quality signal missing (~30K-50K pairs)
- [ ] **Expand synthetic data** — currently 5K pairs, target 50K+
- [ ] Complete Phase 2 SFT run
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
