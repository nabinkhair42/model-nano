"""Training utilities: LR schedules, checkpointing, logging."""

import math
import os
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine LR schedule with linear warmup.

    During the first ``warmup_steps`` the learning rate increases linearly
    from 0 to the optimizer's base LR.  After warmup, it decays following
    a cosine curve down to ``min_lr_ratio * base_lr``.

    Args:
        optimizer: The optimizer whose LR groups will be scheduled.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (warmup + cosine decay).
        min_lr_ratio: Minimum LR as a fraction of the peak LR.

    Returns:
        A ``LambdaLR`` scheduler instance.
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Cosine decay
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save a training checkpoint.

    The checkpoint dict contains:
        - model_state_dict
        - optimizer_state_dict
        - step, epoch, loss

    Args:
        model: The model (or DDP-wrapped model).
        optimizer: The optimizer.
        step: Global training step.
        epoch: Current epoch number.
        loss: Most recent loss value.
        path: File path to write the checkpoint to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP if needed.
    model_to_save = model.module if hasattr(model, "module") else model

    checkpoint = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss,
    }
    # Write to a temp file first then rename for atomicity.
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(path)
    print(f"Checkpoint saved to {path} (step {step}, loss {loss:.4f})")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """Load a checkpoint into a model (and optionally its optimizer).

    Args:
        path: Path to the checkpoint file.
        model: The model to load weights into.
        optimizer: If provided, load optimizer state as well.

    Returns:
        A dict with keys ``step`` and ``epoch`` from the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    # Unwrap DDP if needed.
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
    }


def count_tokens(dataloader: torch.utils.data.DataLoader) -> int:
    """Count total tokens across all batches in a dataloader.

    Assumes each batch's first element is the input_ids tensor with shape
    ``(batch_size, seq_len)``.

    Args:
        dataloader: A DataLoader yielding batches whose first element is
            the input_ids tensor.

    Returns:
        Total number of tokens.
    """
    total = 0
    for batch in dataloader:
        input_ids = batch[0]
        total += input_ids.numel()
    return total


class TrainingLogger:
    """Logs training metrics to console and optionally to Weights & Biases.

    Usage::

        logger = TrainingLogger(use_wandb=True, project="model-nano")
        logger.log_step(step=100, loss=2.3, lr=1e-4, tokens_per_sec=50000)
        logger.log_eval(step=500, val_loss=2.5)
        logger.finish()
    """

    def __init__(
        self,
        use_wandb: bool = False,
        project: str = "model-nano",
        run_name: str = "",
        config: dict | None = None,
    ):
        """
        Args:
            use_wandb: Whether to initialise a W&B run.
            project: W&B project name.
            run_name: W&B run name (auto-generated if empty).
            config: Dict of hyperparameters to log to W&B.
        """
        self.use_wandb = use_wandb
        self._wandb_run = None
        self._start_time = time.time()

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=project,
                    name=run_name or None,
                    config=config or {},
                )
            except ImportError:
                print("WARNING: wandb not installed, falling back to console-only logging.")
                self.use_wandb = False

    def log_step(
        self,
        step: int,
        loss: float,
        lr: float,
        tokens_per_sec: float,
        epoch: int | None = None,
        grad_norm: float | None = None,
        gpu_mem_mb: float | None = None,
    ) -> None:
        """Log a single training step.

        Args:
            step: Global step number.
            loss: Training loss.
            lr: Current learning rate.
            tokens_per_sec: Throughput in tokens/second.
            epoch: Current epoch (optional).
            grad_norm: Gradient norm before clipping (optional).
            gpu_mem_mb: GPU memory allocated in MB (optional).
        """
        elapsed = time.time() - self._start_time
        parts = [f"step {step:>7d}"]
        if epoch is not None:
            parts.append(f"epoch {epoch}")
        parts.append(f"loss {loss:.4f}")
        parts.append(f"lr {lr:.2e}")
        parts.append(f"tok/s {tokens_per_sec:,.0f}")
        if grad_norm is not None:
            parts.append(f"grad_norm {grad_norm:.3f}")
        if gpu_mem_mb is not None:
            parts.append(f"gpu_mem {gpu_mem_mb:,.0f}MB")
        parts.append(f"elapsed {elapsed:.0f}s")

        print(" | ".join(parts))

        if self.use_wandb:
            import wandb
            metrics = {
                "train/loss": loss,
                "train/lr": lr,
                "train/tokens_per_sec": tokens_per_sec,
                "train/elapsed_s": elapsed,
            }
            if epoch is not None:
                metrics["train/epoch"] = epoch
            if grad_norm is not None:
                metrics["train/grad_norm"] = grad_norm
            if gpu_mem_mb is not None:
                metrics["train/gpu_mem_mb"] = gpu_mem_mb
            wandb.log(metrics, step=step)

    def log_eval(self, step: int, val_loss: float) -> None:
        """Log an evaluation result.

        Args:
            step: Global step number.
            val_loss: Validation loss.
        """
        print(f">>> EVAL step {step:>7d} | val_loss {val_loss:.4f}")

        if self.use_wandb:
            import wandb
            wandb.log({"eval/val_loss": val_loss}, step=step)

    def finish(self) -> None:
        """Finalise logging (close W&B run if active)."""
        elapsed = time.time() - self._start_time
        print(f"Training finished in {elapsed:.1f}s")
        if self.use_wandb:
            import wandb
            wandb.finish()
