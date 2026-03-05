"""Phase 1: Pre-training via next-token prediction on documentation corpus.

Usage:
    python -m training.train_pretrain --data-dir data --checkpoint-dir checkpoints/pretrain

All hyperparameters default to the values in ``config.TrainingConfig``.
Any of them can be overridden from the command line.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Ensure project root is on sys.path so bare ``import config`` works.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import ModelConfig, TrainingConfig, DataConfig
from model import NanoGPT
from training.dataset import PretrainDataset, create_dataloader
from training.utils import (
    get_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
    count_tokens,
    TrainingLogger,
)


# ---------------------------------------------------------------------------
# Gradient-checkpointed forward wrapper
# ---------------------------------------------------------------------------

def _block_forward(block, x, rope_freqs, mask):
    """Wrapper that calls a TransformerBlock without KV-cache (training)."""
    out, _ = block(x, rope_freqs, mask, start_pos=0, kv_cache=None)
    return out


def forward_with_gradient_checkpointing(model, input_ids, targets):
    """Run the model forward with gradient checkpointing on each block.

    During training we never use the KV-cache so the per-block signature
    is simplified.  ``torch.utils.checkpoint.checkpoint`` recomputes the
    block activations during the backward pass to save memory.
    """
    batch, seq_len = input_ids.shape
    x = model.tok_emb(input_ids)

    # Causal mask
    mask = None
    if seq_len > 1:
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype
        )
        mask = torch.triu(mask, diagonal=1)

    rope_freqs = model.rope_freqs

    for layer in model.layers:
        x = gradient_checkpoint(
            _block_forward, layer, x, rope_freqs, mask, use_reentrant=False
        )

    x = model.norm(x)
    logits = model.lm_head(x)

    loss = None
    if targets is not None:
        loss = nn.functional.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            targets.view(-1),
            ignore_index=-100,
        )
    return logits, loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, dtype_ctx):
    """Run the model on the validation set and return the mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        input_ids = batch[0].to(device)
        targets = batch[1].to(device)
        with dtype_ctx:
            _, loss, _ = model(input_ids, targets=targets)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    if n_batches == 0:
        return float("nan")
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Configs ----------------------------------------------------------
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()

    # Override from CLI (before auto-config)
    if args.lr is not None:
        train_cfg.lr = args.lr
    if args.min_lr is not None:
        train_cfg.min_lr = args.min_lr
    if args.epochs is not None:
        train_cfg.max_epochs = args.epochs
    if args.micro_batch_size is not None:
        train_cfg.micro_batch_size = args.micro_batch_size
    if args.grad_accumulation_steps is not None:
        train_cfg.grad_accumulation_steps = args.grad_accumulation_steps
    if args.max_grad_norm is not None:
        train_cfg.max_grad_norm = args.max_grad_norm
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.warmup_ratio is not None:
        train_cfg.warmup_ratio = args.warmup_ratio
    if args.log_interval is not None:
        train_cfg.log_interval = args.log_interval
    if args.eval_interval is not None:
        train_cfg.eval_interval = args.eval_interval
    if args.save_interval is not None:
        train_cfg.save_interval = args.save_interval
    if args.wandb:
        train_cfg.use_wandb = True
    if args.wandb_project is not None:
        train_cfg.wandb_project = args.wandb_project

    # Auto-configure batch size and dtype if not specified
    if train_cfg.micro_batch_size == -1 or train_cfg.dtype == "auto":
        train_cfg = train_cfg.auto_configure(model_cfg)

    checkpoint_dir = Path(args.checkpoint_dir or train_cfg.checkpoint_dir)
    data_dir = Path(args.data_dir)

    # ---- Device & precision -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_cfg.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif train_cfg.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    dtype_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32))
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    # ---- Data -------------------------------------------------------------
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        sys.exit(1)

    train_dataset = PretrainDataset(str(train_path), max_seq_len=model_cfg.max_seq_len)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_path.exists():
        val_dataset = PretrainDataset(str(val_path), max_seq_len=model_cfg.max_seq_len)
        val_loader = create_dataloader(
            val_dataset,
            batch_size=train_cfg.micro_batch_size,
            shuffle=False,
            drop_last=False,
        )

    # ---- Model ------------------------------------------------------------
    model = NanoGPT(model_cfg).to(device)
    param_counts = model.count_parameters()
    print(f"Model parameters: {param_counts['total']:,} total "
          f"({param_counts['embedding']:,} embedding, "
          f"{param_counts['transformer']:,} transformer)")

    # ---- Optimizer --------------------------------------------------------
    # Separate weight-decay and no-decay groups.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=train_cfg.eps,
    )

    # ---- Schedule ---------------------------------------------------------
    steps_per_epoch = len(train_loader) // train_cfg.grad_accumulation_steps
    total_steps = steps_per_epoch * train_cfg.max_epochs
    if train_cfg.max_steps > 0:
        total_steps = min(total_steps, train_cfg.max_steps)
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    min_lr_ratio = train_cfg.min_lr / train_cfg.lr

    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio)

    # ---- Resume from checkpoint -------------------------------------------
    start_step = 0
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            info = load_checkpoint(resume_path, model, optimizer)
            start_step = info["step"]
            start_epoch = info["epoch"]
            # Advance scheduler to the correct step.
            for _ in range(start_step):
                scheduler.step()
            print(f"Resumed from {resume_path} at step {start_step}, epoch {start_epoch}")
        else:
            print(f"WARNING: --resume path {resume_path} not found, training from scratch.")

    # ---- Logger -----------------------------------------------------------
    logger = TrainingLogger(
        use_wandb=train_cfg.use_wandb,
        project=train_cfg.wandb_project,
        run_name=train_cfg.wandb_run or "pretrain",
        config={
            "model": model_cfg.__dict__,
            "training": train_cfg.__dict__,
        },
    )

    # ---- Training ---------------------------------------------------------
    use_grad_ckpt = train_cfg.gradient_checkpointing
    model.train()
    global_step = start_step
    tokens_seen = 0
    best_val_loss = float("inf")

    effective_batch = train_cfg.micro_batch_size * train_cfg.grad_accumulation_steps
    print(f"Effective batch size: {effective_batch} "
          f"({train_cfg.micro_batch_size} micro x {train_cfg.grad_accumulation_steps} accum)", flush=True)
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}", flush=True)
    print(f"Training for {train_cfg.max_epochs} epochs, {steps_per_epoch} steps/epoch", flush=True)
    print(f"Gradient checkpointing: {use_grad_ckpt}", flush=True)
    print(f"Device: {device}, dtype: {train_cfg.dtype}", flush=True)
    print("-" * 80, flush=True)

    step_start_time = time.time()
    accum_loss = 0.0

    for epoch in range(start_epoch, train_cfg.max_epochs):
        for micro_step, batch in enumerate(train_loader):
            input_ids = batch[0].to(device)
            targets = batch[1].to(device)

            with dtype_ctx:
                if use_grad_ckpt:
                    _, loss = forward_with_gradient_checkpointing(model, input_ids, targets)
                else:
                    _, loss, _ = model(input_ids, targets=targets)

                # Scale loss by accumulation steps so the effective gradient
                # is the mean over the full effective batch.
                loss = loss / train_cfg.grad_accumulation_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            tokens_seen += input_ids.numel()

            # ---- Accumulation boundary ------------------------------------
            if (micro_step + 1) % train_cfg.grad_accumulation_steps != 0:
                continue

            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), train_cfg.max_grad_norm
            ).item()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]

            # ---- Logging --------------------------------------------------
            if global_step % train_cfg.log_interval == 0:
                elapsed = time.time() - step_start_time
                tok_per_sec = (
                    train_cfg.micro_batch_size
                    * train_cfg.grad_accumulation_steps
                    * model_cfg.max_seq_len
                    * train_cfg.log_interval
                ) / max(elapsed, 1e-9)
                gpu_mem = (
                    torch.cuda.memory_allocated(device) / 1e6
                    if device.type == "cuda"
                    else None
                )
                logger.log_step(
                    step=global_step,
                    loss=accum_loss / train_cfg.log_interval
                    if train_cfg.log_interval > 1
                    else accum_loss,
                    lr=current_lr,
                    tokens_per_sec=tok_per_sec,
                    epoch=epoch,
                    grad_norm=grad_norm,
                    gpu_mem_mb=gpu_mem,
                )
                accum_loss = 0.0
                step_start_time = time.time()

            # ---- Evaluation -----------------------------------------------
            if val_loader is not None and global_step % train_cfg.eval_interval == 0:
                val_loss = evaluate(model, val_loader, device, dtype_ctx)
                logger.log_eval(global_step, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, global_step, epoch, val_loss,
                        checkpoint_dir / "best.pt",
                    )

            # ---- Checkpoint -----------------------------------------------
            if global_step % train_cfg.save_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, epoch, accum_loss,
                    checkpoint_dir / f"step_{global_step}.pt",
                )

            # ---- Early stop on max_steps ----------------------------------
            if train_cfg.max_steps > 0 and global_step >= train_cfg.max_steps:
                break

        if train_cfg.max_steps > 0 and global_step >= train_cfg.max_steps:
            break

    # ---- Final save -------------------------------------------------------
    save_checkpoint(
        model, optimizer, global_step, epoch, accum_loss,
        checkpoint_dir / "final.pt",
    )
    logger.finish()
    print(f"Pre-training complete. Total steps: {global_step}, tokens: {tokens_seen:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Pre-train NanoGPT on a documentation corpus."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing train.bin and val.bin (default: data)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory to save checkpoints (default: from TrainingConfig)",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging.")
    parser.add_argument("--wandb-project", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
