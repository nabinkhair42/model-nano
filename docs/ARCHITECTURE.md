# Architecture

Technical deep-dive into model-nano's design decisions and implementation.

---

## Model Architecture

### Overview

```
Input tokens
    |
    v
[Token Embedding] (16384 x 384)
    |
    v
[TransformerBlock x 32]
    |-- RMSNorm
    |-- GroupedQueryAttention (8 heads, 4 KV heads)
    |-- Residual connection
    |-- RMSNorm
    |-- SwiGLU FFN (384 -> 1024 -> 384)
    |-- Residual connection
    |
    v
[RMSNorm]
    |
    v
[LM Head] (384 -> 16384, weight-tied with embedding)
    |
    v
Logits
```

### Parameter Breakdown

```
Component                  Parameters      % of Total
─────────────────────────────────────────────────────
Token embedding            6,291,456       10.8%
  (16,384 x 384)

Per transformer layer:     1,622,784
  Attention Q projection     147,456
  Attention K projection      73,728
  Attention V projection      73,728
  Attention O projection     147,456
  FFN gate (w1)              393,216
  FFN value (v)              393,216
  FFN down (w2)              393,216
  RMSNorm (x2)                  768

32 layers total           51,929,088       89.2%

Final RMSNorm                   384        0.0%
LM Head                   (tied)           -
─────────────────────────────────────────────────────
Total                     58,220,928       100%
```

### Why These Dimensions?

**32 layers x 384 hidden**: The HuggingFace Dhara-70M study found that 32-layer models score 38.5% vs 12-layer models at 38.15% on benchmarks. Models with <512 hidden AND 16-24 layers fall into a "dead zone" where neither depth nor width is sufficient. Our 32x384 config avoids this by maximizing depth.

**SwiGLU FFN (1024 intermediate)**: LLaMA-style gated FFN. The standard formula for intermediate size is `2/3 * 4 * d = 8/3 * d`, which gives 1024 for d=384. This uses 3 matrices instead of the usual 2, adding ~50% more FFN parameters but providing higher expressivity through the gating mechanism.

**GQA (8Q, 4KV)**: Grouped-Query Attention with 2 query heads sharing each KV head. This halves the KV-cache memory during inference with minimal quality impact. Especially important for fast autoregressive generation.

**16,384 vocab**: At 384 hidden dim, the embedding table is 6.3M params (10.8% of total). A 50K vocab would be 19.2M params (33%) — far too much parameter budget wasted on embeddings for a small model.

**512 context**: Git commands and explanations rarely exceed 200 tokens. 512 gives plenty of headroom while keeping attention O(n^2) manageable and RoPE frequencies compact.

### RoPE Implementation

Rotary Position Embeddings encode position through rotation in complex space:

```
q_rotated = q * e^(i * m * theta)
k_rotated = k * e^(i * n * theta)
```

Where m, n are positions and theta frequencies are:
```
theta_j = 10000^(-2j/d) for j = 0, 1, ..., d/2 - 1
```

This means attention between positions m and n depends only on their relative distance (m - n), giving the model length generalization beyond training sequences.

Frequencies are precomputed once and registered as a non-persistent buffer (not saved in checkpoints).

### Weight Initialization

- All linear layers: N(0, 0.02)
- Embedding: N(0, 0.02)
- Residual projections (attention output, FFN down): N(0, 0.02 / sqrt(2 * n_layers))
  - This scaling prevents residual stream variance from growing with depth (GPT-2 paper)
- Weight tying: LM head shares the embedding weight matrix

---

## Training Architecture

### Two-Phase Approach

**Phase 1 — Pre-training (next-token prediction)**

The model learns git vocabulary, syntax patterns, and language structure by predicting the next token in documentation text. This phase uses all raw text (Pro Git, man pages, tutorials) without any instruction formatting.

- Full cross-entropy loss on all tokens
- Higher learning rate (1e-3)
- Trains for 10-20 epochs or until validation loss plateaus

**Phase 2 — Supervised Fine-tuning (SFT)**

The model learns to follow instructions by training on ChatML-formatted instruction-response pairs. Only the assistant's response tokens contribute to the loss.

- Loss masking: system/user tokens have loss weight 0
- Lower learning rate (2e-5) per the "Secret Recipe" paper
- Trains for 3-5 epochs
- Loads Phase 1 weights as starting point

### Memory Optimization

Three techniques make training possible on 4GB VRAM:

1. **BF16 mixed precision**: Halves memory for weights, gradients, and activations. BFloat16 has the same exponent range as FP32 (no loss scaling needed), unlike FP16.

2. **Gradient checkpointing**: Instead of storing all intermediate activations, recompute them during the backward pass. Trades ~30% extra compute for ~60% memory savings. **Required for GPUs with ≤6GB VRAM.** Can be disabled with `--no-gradient-checkpointing` for faster training on larger GPUs.

3. **Gradient accumulation**: Process micro-batches of 4 sequences, accumulate gradients over 16 steps, then update. Achieves effective batch size of 64 without needing memory for 64 sequences.

### Memory Breakdown (4GB GPU)

| Component | Size |
|-----------|------|
| Model weights (BF16) | ~116 MB |
| Optimizer states (FP32) | ~465 MB |
| Gradients (BF16) | ~116 MB |
| CUDA overhead | ~200 MB |
| **Total baseline** | **~900 MB** |

With gradient checkpointing enabled, activation memory is minimal. Disabling it on a 4GB GPU causes OOM.

### Optimizer Details

AdamW with decoupled weight decay:
- Only 2D+ parameters get weight decay (linear layers)
- 1D parameters (norms, biases) have weight_decay=0
- This prevents regularizing scale parameters which hurts training

---

## Inference Architecture

### KV-Cache

During autoregressive generation, each new token only needs to attend to all previous tokens. Without caching, this requires recomputing all key/value projections for the entire sequence at each step — O(n^2) total work.

With KV-cache:
1. **Prefill**: Run the full prompt through the model once, cache all K/V tensors per layer
2. **Decode**: For each new token, only compute Q/K/V for that one token, concatenate K/V with cache

This reduces generation from O(n^2) to O(n), enabling 500-2000+ tokens/sec on an RTX 3050.

Cache memory per layer: `2 * batch * seq_len * n_kv_heads * head_dim * 2 bytes`
Total for 512 tokens: `2 * 1 * 512 * 4 * 48 * 2 * 32 layers = ~12.6 MB` (negligible)

### Sampling Strategies

| Use Case | Strategy | Settings |
|----------|----------|----------|
| Commands | Greedy | temperature=0 (deterministic) |
| Explanations | Nucleus | temperature=0.7, top_p=0.9 |
| Completions | Top-k | temperature=0.8, top_k=10, n samples |

### Export Paths

- **PyTorch (.pt)**: Primary format, full model with KV-cache support
- **ONNX + INT8**: For CPU deployment, ~4x smaller, uses dynamic quantization
- **GGUF**: Future — for llama.cpp ecosystem compatibility

---

## Data Architecture

### ChatML Format

All instruction data uses OpenAI's ChatML format:

```
<|im_start|>system
You are a Git expert. Provide precise, correct git commands and explanations.<|im_end|>
<|im_start|>user
How do I undo my last commit but keep the changes staged?<|im_end|>
<|im_start|>assistant
git reset --soft HEAD~1

This moves HEAD back one commit while keeping all changes in the staging area.<|im_end|>
```

Special tokens:
- `<|im_start|>` (id=0): Start of a message role block
- `<|im_end|>` (id=1): End of a message role block
- `<|pad|>` (id=2): Padding token

### Task Distribution

Target balance across training data:
- 30% command generation
- 20% explanation / Q&A
- 15% error diagnosis
- 15% multi-step workflows
- 10% flag/option reference
- 10% GitHub CLI operations

### Deduplication

The prepare_dataset pipeline deduplicates by normalizing text (lowercase, collapse whitespace) and removing exact matches. This is a simple but effective strategy that removed ~40% duplicates in testing.

---

## CLI Architecture

### Suggest-Then-Confirm Pattern

```
$ git-nano "undo last 3 commits but keep changes"

  git reset --soft HEAD~3

  Resets the last 3 commits while keeping all changes staged.

  [Enter] Execute  [e] Edit  [c] Copy  [q] Cancel
```

This pattern is safe by default — the user must explicitly confirm before any command runs. Destructive commands (force-push, reset --hard) get an extra warning.

### Context Awareness

The CLI auto-detects:
- Current branch name
- File status (modified, staged, untracked counts)
- Last 5 commit messages
- Remote names

This context is injected into the prompt so the model can give branch-specific advice.

### Modes

1. **One-shot**: `git-nano "query"` — answer and exit
2. **Interactive**: `git-nano` — REPL with conversation history
3. **Pipe**: `git status | git-nano explain` — process piped input
4. **Subcommand**: `git nano "query"` via git alias

---

## Evaluation Architecture

### Metrics

**Exact match**: Normalize (strip, lowercase, collapse whitespace) and compare against all acceptable alternatives.

**Command equivalence**: Parse commands into (base, flags, args) and compare semantically. Handles:
- Flag aliases: `--staged` = `--cached`, `-n 5` = `-5`
- Short/long flags: `-a` = `--all`
- Command synonyms: `git checkout -b` = `git switch -c`
- Combined short flags: `-am` = `-a -m`

**Response quality**: Heuristic scoring (0-1) for explanations:
- 60% weight: term overlap with reference answer
- 20% weight: length appropriateness
- 20% weight: structural quality (code blocks, lists)

### Test Coverage

50 test cases across 8 categories:
- basic (10), branching (8), history (8), remote (6)
- stash (4), config (4), error_recovery (6), github_cli (4)

Target: expand to 300 test cases for comprehensive evaluation.
