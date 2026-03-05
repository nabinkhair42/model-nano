#!/usr/bin/env python
"""One-command training script for model-nano.

This script handles the entire training pipeline:
1. Generate synthetic data
2. Prepare datasets
3. Pre-train on documentation
4. Fine-tune on instructions

Usage:
    python train.py                    # Full pipeline
    python train.py --skip-data        # Skip data generation (if already done)
    python train.py --sft-only         # Only run SFT (requires pretrain checkpoint)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    # Use unbuffered output
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False

    print(f"\n✅ COMPLETED: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train model-nano from scratch"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation and preparation (use existing data)",
    )
    parser.add_argument(
        "--sft-only",
        action="store_true",
        help="Only run SFT phase (requires existing pretrain checkpoint)",
    )
    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=20000,
        help="Number of synthetic examples to generate (default: 20000)",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=10,
        help="Number of pretraining epochs (default: 10)",
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=3,
        help="Number of SFT epochs (default: 3)",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=str,
        default="checkpoints/pretrain/best.pt",
        help="Pretrain checkpoint for SFT (default: checkpoints/pretrain/best.pt)",
    )
    args = parser.parse_args()

    # Ensure we're in the project root
    project_root = Path(__file__).parent
    os.chdir(project_root)

    print("=" * 60)
    print("MODEL-NANO TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Generate synthetic data
    if not args.skip_data and not args.sft_only:
        success = run_command(
            [
                sys.executable, "data/generate_synthetic.py",
                "--count", str(args.synthetic_count),
            ],
            "Generate synthetic training data"
        )
        if not success:
            sys.exit(1)

    # Step 2: Prepare pretraining dataset
    if not args.skip_data and not args.sft_only:
        success = run_command(
            [sys.executable, "data/prepare_dataset.py"],
            "Prepare pretraining dataset"
        )
        if not success:
            sys.exit(1)

    # Step 3: Prepare SFT dataset
    if not args.skip_data:
        success = run_command(
            [sys.executable, "data/prepare_sft.py"],
            "Prepare SFT dataset"
        )
        if not success:
            sys.exit(1)

    # Step 4: Pre-training
    if not args.sft_only:
        success = run_command(
            [
                sys.executable, "-m", "training.train_pretrain",
                "--epochs", str(args.pretrain_epochs),
                "--checkpoint-dir", "checkpoints/pretrain",
            ],
            f"Pre-training ({args.pretrain_epochs} epochs)"
        )
        if not success:
            sys.exit(1)

    # Step 5: SFT
    checkpoint = args.pretrain_checkpoint
    if not Path(checkpoint).exists():
        print(f"\n❌ ERROR: Pretrain checkpoint not found: {checkpoint}")
        print("Run pretraining first or specify --pretrain-checkpoint")
        sys.exit(1)

    success = run_command(
        [
            sys.executable, "-m", "training.train_sft",
            "--pretrain-checkpoint", checkpoint,
            "--epochs", str(args.sft_epochs),
            "--checkpoint-dir", "checkpoints/sft",
        ],
        f"Supervised Fine-Tuning ({args.sft_epochs} epochs)"
    )
    if not success:
        sys.exit(1)

    # Done
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("\nCheckpoints saved to:")
    print("  - Pretrain: checkpoints/pretrain/best.pt")
    print("  - SFT:      checkpoints/sft/best.pt")
    print("\nTo test the model:")
    print("  python -m cli 'how do I undo my last commit'")
    print("=" * 60)


if __name__ == "__main__":
    main()
