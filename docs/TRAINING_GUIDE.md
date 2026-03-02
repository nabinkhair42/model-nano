# Training Guide

Step-by-step instructions for training model-nano from scratch.

---

## Prerequisites

### Hardware
- GPU with 4+ GB VRAM (RTX 3050 or better)
- 16+ GB system RAM
- 10+ GB disk space (for data + checkpoints)

### Software
```bash
pip install -r requirements.txt
# or
pip install -e ".[train]"
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 1: Collect Data

### Real Documentation

```bash
python data/collect_docs.py
```

This clones and parses:
- Pro Git book (AsciiDoc -> plain text)
- Git man pages (168 commands)
- tldr-pages (structured command examples)
- GitHub Docs (workflows, API guides)

Output: `data/raw/docs.jsonl`

Takes ~5-10 minutes (cloning repos).

### Stack Overflow (Optional)

Download the Stack Overflow data dump from Archive.org, then:

```bash
python data/collect_stackoverflow.py --input Posts.xml
```

Filters for `[git]` tagged posts with score >= 5. Output: `data/raw/stackoverflow.jsonl`

### Synthetic Data

```bash
python data/generate_synthetic.py --count 5000
```

Generates instruction pairs using three strategies:
- Seed expansion (command variations)
- Error scenarios (error -> diagnosis + fix)
- Flag combinatorics (command + flag pairs)

Output: `data/raw/synthetic.jsonl`

---

## Step 2: Train Tokenizer

```bash
python tokenizer/train_tokenizer.py
```

Options:
```
--input-dir DIR     Source text directory (default: data/raw)
--output PATH       Output path (default: tokenizer/tokenizer.json)
--vocab-size N      Vocabulary size (default: 16384)
--min-frequency N   Minimum token frequency (default: 2)
```

The tokenizer needs a large enough corpus to reach the full 16,384 vocab. With only synthetic data, it will be smaller (~1,500). With all sources combined, it should reach or approach the target.

Verify:
```bash
python tokenizer/train_tokenizer.py --verify-only
```

---

## Step 3: Prepare Dataset

```bash
python data/prepare_dataset.py
```

This pipeline:
1. Loads all JSONL files from `data/raw/`
2. Deduplicates by normalized content
3. Formats into tokenizable text (ChatML for instructions, raw for docs)
4. Tokenizes using the trained tokenizer
5. Splits into train/val (95/5)
6. Saves as `.bin` files (uint16 numpy memmap)

Output: `data/train.bin`, `data/val.bin`

Options:
```
--raw-dir DIR       Raw data directory (default: data/raw)
--output-dir DIR    Output directory (default: data)
--tokenizer PATH    Tokenizer path (default: tokenizer/tokenizer.json)
--max-seq-len N     Max sequence length (default: 512)
--train-split F     Train split ratio (default: 0.95)
```

---

## Step 4: Phase 1 Pre-training

```bash
python -m training.train_pretrain \
    --data-dir data \
    --checkpoint-dir checkpoints/pretrain \
    --epochs 20
```

### What to Expect

- Initial loss: ~9.7 (= ln(16384), random chance)
- After convergence: loss should drop below 4.0
- Training time: depends heavily on dataset size and GPU
  - 50M tokens on RTX 3050: ~2-4 hours
  - 100M tokens: ~4-8 hours

### Key Options

```
--lr FLOAT                Peak learning rate (default: 1e-3)
--micro-batch-size N      Sequences per GPU step (default: 4)
--grad-accumulation N     Accumulation steps (default: 16)
--epochs N                Max epochs (default: 20)
--max-steps N             Max steps, overrides epochs (default: -1)
--checkpoint-dir DIR      Where to save checkpoints
--resume PATH             Resume from checkpoint
--wandb                   Enable wandb logging
```

### Monitoring

Watch for:
- **Loss decreasing** steadily (not flat, not exploding)
- **Val loss tracking** train loss (not diverging = no overfitting)
- **GPU memory** staying under 3.5 GB (`nvidia-smi`)
- **Gradient norm** staying reasonable (not spiking)

The script logs loss, learning rate, tokens/sec, gradient norm, and GPU memory every 10 steps.

### Checkpoints

- `best.pt` — saved whenever val loss improves
- `step_XXXX.pt` — saved every 1000 steps
- `final.pt` — saved at end of training

Each checkpoint contains model weights, optimizer state, step, and epoch.

### Resuming

```bash
python -m training.train_pretrain --resume checkpoints/pretrain/step_5000.pt
```

---

## Step 5: Phase 2 SFT

```bash
python -m training.train_sft \
    --pretrain-checkpoint checkpoints/pretrain/best.pt \
    --data-dir data \
    --checkpoint-dir checkpoints/sft \
    --epochs 5
```

### Differences from Pre-training

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Objective | Next-token prediction | Instruction following |
| Data | Raw documentation text | ChatML instruction pairs |
| Loss | All tokens | Assistant completions only |
| Learning rate | 1e-3 | 2e-5 |
| Epochs | 10-20 | 3-5 |

### Loss Masking

The SFT training masks loss on system and user tokens. Only the assistant's response contributes to the gradient. This focuses the model on generating good responses rather than memorizing prompts.

### Key Options

```
--pretrain-checkpoint PATH    Phase 1 weights to load (required)
--lr FLOAT                    Peak LR (default: 2e-5)
--no-mask-prompt              Disable loss masking (train on all tokens)
```

### What to Expect

- Initial loss: lower than Phase 1 start (model already knows git vocabulary)
- SFT loss should decrease quickly (3-5 epochs is usually enough)
- Watch for val loss diverging — stop early if overfitting

---

## Step 6: Evaluation

```bash
python -m eval.benchmark \
    --model checkpoints/sft/best.pt \
    --verbose \
    --show-failures
```

This runs the 50 test cases and reports per-category accuracy:
- **Command tasks**: scored by exact match + command equivalence
- **Explanation tasks**: scored by response quality heuristics

Target accuracy: 50-70% on v1 (this is a small model — it won't match GPT-4).

---

## Troubleshooting

### Out of Memory

If you get CUDA OOM errors:

1. **Reduce micro batch size**: `--micro-batch-size 2` (increase `--grad-accumulation` to compensate)
2. **Verify gradient checkpointing**: Should be enabled by default
3. **Check other GPU processes**: `nvidia-smi` — kill anything using VRAM
4. **Reduce sequence length**: Edit `config.py`, set `max_seq_len = 256`

### Loss Not Decreasing

1. **Learning rate too high**: Try `--lr 3e-4` for Phase 1
2. **Data too small**: Need at least 10M tokens for Phase 1
3. **Data quality**: Check that tokenized data decodes correctly
4. **Gradient norm**: If spiking, reduce `--lr`

### NaN Loss

1. **Switch to FP32**: Edit training config, set `dtype = "float32"`
2. **Lower learning rate**: `--lr 1e-4`
3. **Check data**: Look for corrupted tokens or empty sequences

### Slow Training

1. **Enable CUDA**: Verify `torch.cuda.is_available()` returns True
2. **Pin memory**: Enabled by default in DataLoader
3. **Persistent workers**: Enabled by default
4. **Compile model**: Add `torch.compile(model)` in training script (PyTorch 2.0+)
