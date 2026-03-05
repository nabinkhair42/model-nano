# Model-Nano: Git-Specialized Developer LLM

A ~58M parameter transformer LLM trained from scratch, specialized for Git/GitHub developer assistance. The model handles three core tasks:

1. **Command Generation**: Natural language → precise git commands
2. **Explanation**: Explaining git concepts and workflows
3. **Autocomplete**: Completing partial git commands

## Features

- **Auto GPU Detection**: Automatically configures batch size and precision based on available GPU memory
- **Universal Compatibility**: Runs on any CUDA GPU (4GB+), Apple Silicon, or CPU
- **Efficient Training**: Gradient checkpointing, mixed precision (BF16/FP16), and KV-cache inference
- **ChatML Format**: Standard conversation format for easy integration

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/model-nano.git
cd model-nano

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training, optional)

### GPU Compatibility

| GPU | VRAM | Batch Size | Precision | Status |
|-----|------|------------|-----------|--------|
| RTX 4090 | 24GB | 16+ | BF16 | ✅ Excellent |
| RTX 3090 | 24GB | 16+ | BF16 | ✅ Excellent |
| RTX 3080 | 10GB | 8 | BF16 | ✅ Great |
| RTX 3070 | 8GB | 6 | BF16 | ✅ Good |
| RTX 3060 | 12GB | 8 | BF16 | ✅ Great |
| RTX 3050 | 4GB | 2-4 | BF16 | ✅ Works |
| GTX 1080 Ti | 11GB | 6 | FP16 | ✅ Good |
| GTX 1070 | 8GB | 4 | FP16 | ✅ Good |
| Apple M1/M2 | 8GB+ | 4 | FP32 | ✅ Works |
| CPU | - | 1 | FP32 | ⚠️ Slow |

## Training

### Full Pipeline (Recommended)

```bash
# 1. Generate synthetic training data
python data/generate_synthetic.py --count 20000

# 2. Prepare datasets
python data/prepare_dataset.py
python data/prepare_sft.py

# 3. Pre-train on documentation (Phase 1)
python -m training.train_pretrain --epochs 10

# 4. Fine-tune on instructions (Phase 2)
python -m training.train_sft --pretrain-checkpoint checkpoints/best.pt --epochs 3
```

### Auto GPU Configuration

Training automatically detects your GPU and configures optimal settings:

```bash
# Auto-detect everything (recommended)
python -m training.train_pretrain

# The script will output:
# ============================================================
# GPU CONFIGURATION
# ============================================================
# Device:              NVIDIA GeForce RTX 3050
# Total Memory:        4.00 GB
# Available Memory:    3.50 GB
# Compute Capability:  8.6
# BF16 Support:        Yes
# Optimal Precision:   bfloat16
# ============================================================
#
# AUTO-CONFIGURED TRAINING:
#   Micro batch size:     4
#   Grad accumulation:    16
#   Effective batch:      64
#   Precision:            bfloat16
```

### Manual Configuration

Override auto-detection with command-line arguments:

```bash
# Force specific batch size
python -m training.train_pretrain --micro-batch-size 8 --grad-accumulation-steps 8

# Use FP16 instead of BF16
python -m training.train_pretrain --dtype float16
```

### Check GPU Info

```bash
python -m training.gpu_utils
```

## Inference

### Interactive CLI

```bash
# Start interactive mode
python -m cli

# Or one-shot query
python -m cli "how do I undo my last commit"
```

### Python API

```python
from inference.engine import InferenceEngine

engine = InferenceEngine(
    model_path="checkpoints/sft/best.pt",
    tokenizer_path="tokenizer/tokenizer.json",
)

# Generate a command
prompt = engine.format_prompt("How do I undo my last commit but keep the changes?")
response = engine.generate(prompt, max_new_tokens=256, temperature=0.0)
print(response)
# Output: git reset --soft HEAD~1
```

## Project Structure

```
model-nano/
├── config.py                 # Model + training configuration (auto-GPU)
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── generate_synthetic.py # Generate training data
│   ├── prepare_dataset.py    # Prepare pretraining data
│   ├── prepare_sft.py        # Prepare SFT data with loss masks
│   └── raw/                  # Raw data sources
│
├── tokenizer/
│   ├── train_tokenizer.py    # Train BPE tokenizer
│   └── tokenizer.json        # Trained tokenizer
│
├── model/
│   ├── transformer.py        # Full transformer model
│   ├── attention.py          # GQA attention with KV-cache
│   └── components.py         # RMSNorm, SwiGLU, RoPE
│
├── training/
│   ├── train_pretrain.py     # Phase 1: Pretraining
│   ├── train_sft.py          # Phase 2: SFT
│   ├── dataset.py            # PyTorch datasets
│   ├── gpu_utils.py          # Auto GPU detection
│   └── utils.py              # Training utilities
│
├── inference/
│   ├── engine.py             # Inference engine with KV-cache
│   └── generate.py           # Sampling strategies
│
├── cli/
│   ├── __main__.py           # CLI entry point
│   └── oneshot.py            # One-shot query mode
│
└── checkpoints/              # Saved model checkpoints
```

## Model Architecture

| Component | Value |
|-----------|-------|
| Parameters | ~58M |
| Layers | 32 |
| Hidden dim | 384 |
| FFN dim | 1,024 |
| Attention heads | 8 |
| KV heads | 4 (GQA) |
| Context length | 512 tokens |
| Vocab size | 16,384 |
| Position encoding | RoPE |
| Normalization | RMSNorm (pre-norm) |
| Activation | SwiGLU |

## Training Data

The model is trained on:
- **Real data**: Pro Git book, git man pages, tldr-pages, GitHub CLI docs
- **Synthetic data**: Generated command variations, error scenarios, flag combinations

## Tips for Best Results

1. **Use temperature=0.0** for command generation (deterministic output)
2. **Use temperature=0.7** for explanations (more natural language)
3. **Include context** when available (current branch, status)
4. **Be specific** in your queries for better results

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors, reduce batch size:

```bash
python -m training.train_pretrain --micro-batch-size 2
```

### Slow Training on GPU

Ensure CUDA is properly installed:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Outputs Garbage

1. Ensure you're using the SFT checkpoint (not pretrain)
2. Use correct ChatML format: `engine.format_prompt(query)`
3. Use `temperature=0.0` for commands

## License

MIT License - see LICENSE file for details.

## Citation

If you use this model in your research, please cite:

```bibtex
@software{model_nano,
  title = {Model-Nano: A Git-Specialized Developer LLM},
  year = {2024},
  url = {https://github.com/yourusername/model-nano}
}
```
