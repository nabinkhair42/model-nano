"""Phase 2: Supervised Fine-Tuning on instruction-response pairs.

Loads a Phase 1 pre-trained checkpoint and continues training with a
lower learning rate on ChatML-formatted instruction data.  Only
assistant-completion tokens contribute to the loss (prompt masking).

Usage:
    python -m training.train_sft \
        --pretrain-checkpoint checkpoints/pretrain/final.pt \
        --data-dir data/sft \
        --checkpoint-dir checkpoints/sft
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import ModelConfig, SFTConfig
from model import NanoGPT
from training.dataset import SFTDataset, create_dataloader
from training.utils import (
    get_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
    count_tokens,
    TrainingLogger,
)


# ---------------------------------------------------------------------------
# Gradient-checkpointed forward with loss masking
# ---------------------------------------------------------------------------

def _block_forward(block, x, rope_freqs, mask):
    """Wrapper that calls a TransformerBlock without KV-cache (training)."""
    out, _ = block(x, rope_freqs, mask, start_pos=0, kv_cache=None)
    return out


def forward_with_gradient_checkpointing(model, input_ids, targets, loss_mask=None):
    """Model forward with gradient checkpointing and optional loss masking.

    When ``loss_mask`` is provided (shape ``(batch, seq_len)``, float),
    the per-token cross-entropy is multiplied element-wise by the mask
    and then averaged over the unmasked tokens.
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
        if loss_mask is not None:
            # Per-token loss with masking
            per_token_loss = nn.functional.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
                reduction="none",
            )
            per_token_loss = per_token_loss.view(batch, seq_len)
            masked_loss = per_token_loss * loss_mask
            # Average over unmasked tokens only
            n_tokens = loss_mask.sum()
            loss = masked_loss.sum() / n_tokens.clamp(min=1)
        else:
            loss = nn.functional.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
    return logits, loss


def forward_no_checkpointing(model, input_ids, targets, loss_mask=None):
    """Standard forward with optional loss masking (no gradient checkpointing)."""
    logits, ce_loss, _ = model(input_ids, targets=targets)

    if loss_mask is not None and targets is not None:
        batch, seq_len = input_ids.shape
        per_token_loss = nn.functional.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            targets.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        per_token_loss = per_token_loss.view(batch, seq_len)
        masked_loss = per_token_loss * loss_mask
        n_tokens = loss_mask.sum()
        loss = masked_loss.sum() / n_tokens.clamp(min=1)
        return logits, loss

    return logits, ce_loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device, dtype_ctx, mask_prompt=True):
    """Run the model on the SFT validation set and return the mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in val_loader:
        input_ids = batch[0].to(device)
        targets = batch[1].to(device)
        loss_mask = batch[2].to(device) if mask_prompt else None

        with dtype_ctx:
            _, loss = forward_no_checkpointing(model, input_ids, targets, loss_mask)

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
    sft_cfg = SFTConfig()

    # Override from CLI
    if args.lr is not None:
        sft_cfg.lr = args.lr
    if args.min_lr is not None:
        sft_cfg.min_lr = args.min_lr
    if args.epochs is not None:
        sft_cfg.max_epochs = args.epochs
    if args.micro_batch_size is not None:
        sft_cfg.micro_batch_size = args.micro_batch_size
    if args.grad_accumulation_steps is not None:
        sft_cfg.grad_accumulation_steps = args.grad_accumulation_steps
    if args.max_grad_norm is not None:
        sft_cfg.max_grad_norm = args.max_grad_norm
    if args.weight_decay is not None:
        sft_cfg.weight_decay = args.weight_decay
    if args.warmup_ratio is not None:
        sft_cfg.warmup_ratio = args.warmup_ratio
    if args.log_interval is not None:
        sft_cfg.log_interval = args.log_interval
    if args.eval_interval is not None:
        sft_cfg.eval_interval = args.eval_interval
    if args.save_interval is not None:
        sft_cfg.save_interval = args.save_interval
    if args.wandb:
        sft_cfg.use_wandb = True
    if args.wandb_project is not None:
        sft_cfg.wandb_project = args.wandb_project
    if args.no_mask_prompt:
        sft_cfg.mask_prompt = False

    checkpoint_dir = Path(args.checkpoint_dir or sft_cfg.checkpoint_dir)
    data_dir = Path(args.data_dir)

    # ---- Device & precision -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if sft_cfg.dtype == "bfloat16" else torch.float16
    dtype_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    # ---- Data -------------------------------------------------------------
    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    if not train_path.exists():
        print(f"ERROR: SFT training data not found at {train_path}")
        sys.exit(1)

    train_dataset = SFTDataset(
        str(train_path),
        max_seq_len=model_cfg.max_seq_len,
        mask_prompt=sft_cfg.mask_prompt,
    )
    train_loader = create_dataloader(
        train_dataset,
        batch_size=sft_cfg.micro_batch_size,
        shuffle=True,
    )

    val_loader = None
    if val_path.exists():
        val_dataset = SFTDataset(
            str(val_path),
            max_seq_len=model_cfg.max_seq_len,
            mask_prompt=sft_cfg.mask_prompt,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=sft_cfg.micro_batch_size,
            shuffle=False,
            drop_last=False,
        )

    # ---- Model ------------------------------------------------------------
    model = NanoGPT(model_cfg).to(device)
    param_counts = model.count_parameters()
    print(f"Model parameters: {param_counts['total']:,} total")

    # ---- Load pretrained weights ------------------------------------------
    if args.pretrain_checkpoint:
        pretrain_path = Path(args.pretrain_checkpoint)
        if not pretrain_path.exists():
            print(f"ERROR: Pre-train checkpoint not found at {pretrain_path}")
            sys.exit(1)
        # Load model weights only (not optimizer state).
        info = load_checkpoint(pretrain_path, model, optimizer=None)
        print(f"Loaded pre-trained weights from {pretrain_path} "
              f"(pretrain step {info['step']}, epoch {info['epoch']})")
    else:
        print("WARNING: No --pretrain-checkpoint provided. Training SFT from scratch.")

    # ---- Optimizer --------------------------------------------------------
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
            {"params": decay_params, "weight_decay": sft_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=sft_cfg.lr,
        betas=(sft_cfg.beta1, sft_cfg.beta2),
        eps=sft_cfg.eps,
    )

    # ---- Schedule ---------------------------------------------------------
    steps_per_epoch = len(train_loader) // sft_cfg.grad_accumulation_steps
    total_steps = steps_per_epoch * sft_cfg.max_epochs
    if sft_cfg.max_steps > 0:
        total_steps = min(total_steps, sft_cfg.max_steps)
    warmup_steps = int(total_steps * sft_cfg.warmup_ratio)
    min_lr_ratio = sft_cfg.min_lr / sft_cfg.lr

    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps, min_lr_ratio)

    # ---- Resume from SFT checkpoint --------------------------------------
    start_step = 0
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            info = load_checkpoint(resume_path, model, optimizer)
            start_step = info["step"]
            start_epoch = info["epoch"]
            for _ in range(start_step):
                scheduler.step()
            print(f"Resumed SFT from {resume_path} at step {start_step}, epoch {start_epoch}")
        else:
            print(f"WARNING: --resume path {resume_path} not found, starting SFT from loaded weights.")

    # ---- Logger -----------------------------------------------------------
    logger = TrainingLogger(
        use_wandb=sft_cfg.use_wandb,
        project=sft_cfg.wandb_project,
        run_name=sft_cfg.wandb_run or "sft",
        config={
            "model": model_cfg.__dict__,
            "sft": sft_cfg.__dict__,
        },
    )

    # ---- Training ---------------------------------------------------------
    use_grad_ckpt = sft_cfg.gradient_checkpointing
    mask_prompt = sft_cfg.mask_prompt
    model.train()
    global_step = start_step
    tokens_seen = 0
    best_val_loss = float("inf")

    effective_batch = sft_cfg.micro_batch_size * sft_cfg.grad_accumulation_steps
    print(f"Effective batch size: {effective_batch}")
    print(f"Total steps: {total_steps}, warmup: {warmup_steps}")
    print(f"SFT for {sft_cfg.max_epochs} epochs, {steps_per_epoch} steps/epoch")
    print(f"Loss masking (train on completions only): {mask_prompt}")
    print(f"LR: {sft_cfg.lr}, min_lr: {sft_cfg.min_lr}")
    print(f"Gradient checkpointing: {use_grad_ckpt}")
    print(f"Device: {device}, dtype: {sft_cfg.dtype}")
    print("-" * 80)

    step_start_time = time.time()
    accum_loss = 0.0

    for epoch in range(start_epoch, sft_cfg.max_epochs):
        for micro_step, batch in enumerate(train_loader):
            input_ids = batch[0].to(device)
            targets = batch[1].to(device)
            loss_mask = batch[2].to(device) if mask_prompt else None

            with dtype_ctx:
                if use_grad_ckpt:
                    _, loss = forward_with_gradient_checkpointing(
                        model, input_ids, targets, loss_mask
                    )
                else:
                    _, loss = forward_no_checkpointing(
                        model, input_ids, targets, loss_mask
                    )

                loss = loss / sft_cfg.grad_accumulation_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            tokens_seen += input_ids.numel()

            # ---- Accumulation boundary ------------------------------------
            if (micro_step + 1) % sft_cfg.grad_accumulation_steps != 0:
                continue

            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(
                model.parameters(), sft_cfg.max_grad_norm
            ).item()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            global_step += 1
            current_lr = scheduler.get_last_lr()[0]

            # ---- Logging --------------------------------------------------
            if global_step % sft_cfg.log_interval == 0:
                elapsed = time.time() - step_start_time
                tok_per_sec = (
                    sft_cfg.micro_batch_size
                    * sft_cfg.grad_accumulation_steps
                    * model_cfg.max_seq_len
                    * sft_cfg.log_interval
                ) / max(elapsed, 1e-9)
                gpu_mem = (
                    torch.cuda.memory_allocated(device) / 1e6
                    if device.type == "cuda"
                    else None
                )
                logger.log_step(
                    step=global_step,
                    loss=accum_loss / sft_cfg.log_interval
                    if sft_cfg.log_interval > 1
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
            if val_loader is not None and global_step % sft_cfg.eval_interval == 0:
                val_loss = evaluate(
                    model, val_loader, device, dtype_ctx, mask_prompt
                )
                logger.log_eval(global_step, val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, global_step, epoch, val_loss,
                        checkpoint_dir / "best.pt",
                    )

            # ---- Checkpoint -----------------------------------------------
            if global_step % sft_cfg.save_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, epoch, accum_loss,
                    checkpoint_dir / f"step_{global_step}.pt",
                )

            # ---- Early stop -----------------------------------------------
            if sft_cfg.max_steps > 0 and global_step >= sft_cfg.max_steps:
                break

        if sft_cfg.max_steps > 0 and global_step >= sft_cfg.max_steps:
            break

    # ---- Final save -------------------------------------------------------
    save_checkpoint(
        model, optimizer, global_step, epoch, accum_loss,
        checkpoint_dir / "final.pt",
    )
    logger.finish()
    print(f"SFT complete. Total steps: {global_step}, tokens: {tokens_seen:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2: Supervised fine-tuning of NanoGPT."
    )
    parser.add_argument(
        "--pretrain-checkpoint", type=str, default=None,
        help="Path to Phase 1 pretrained checkpoint (required for proper SFT).",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/sft",
        help="Directory containing SFT train.bin / val.bin (default: data/sft)",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Directory to save SFT checkpoints.",
    )
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to SFT checkpoint to resume from.")
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
    parser.add_argument("--no-mask-prompt", action="store_true",
                        help="Disable prompt masking (train on full sequence).")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
