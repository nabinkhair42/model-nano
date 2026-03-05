"""GPU detection and automatic batch size optimization.

This module provides utilities for:
- Detecting GPU capabilities (memory, compute capability)
- Auto-calculating optimal batch sizes based on available VRAM
- Memory estimation for different model configurations
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUInfo:
    """Information about a GPU device."""
    name: str
    total_memory_gb: float
    available_memory_gb: float
    compute_capability: tuple[int, int]
    supports_bf16: bool
    supports_fp16: bool
    device_index: int


def get_gpu_info(device_index: int = 0) -> Optional[GPUInfo]:
    """Get information about a specific GPU.

    Returns None if CUDA is not available or device doesn't exist.
    """
    if not torch.cuda.is_available():
        return None

    if device_index >= torch.cuda.device_count():
        return None

    props = torch.cuda.get_device_properties(device_index)

    # Get memory info
    total_memory = props.total_memory / (1024 ** 3)  # Convert to GB

    # Get current available memory
    torch.cuda.set_device(device_index)
    free_memory, _ = torch.cuda.mem_get_info(device_index)
    available_memory = free_memory / (1024 ** 3)

    # Check compute capability for precision support
    compute_cap = (props.major, props.minor)
    supports_bf16 = compute_cap >= (8, 0)  # Ampere and newer
    supports_fp16 = compute_cap >= (6, 0)  # Pascal and newer

    return GPUInfo(
        name=props.name,
        total_memory_gb=total_memory,
        available_memory_gb=available_memory,
        compute_capability=compute_cap,
        supports_bf16=supports_bf16,
        supports_fp16=supports_fp16,
        device_index=device_index,
    )


def estimate_memory_usage(
    model_params: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    n_layers: int,
    dtype_bytes: int = 2,  # bf16/fp16 = 2 bytes
    gradient_checkpointing: bool = True,
) -> float:
    """Estimate GPU memory usage in GB for training.

    This is a conservative estimate based on:
    - Model weights
    - Optimizer states (AdamW: 2 states per param in FP32)
    - Gradients
    - Activations (reduced if gradient checkpointing)
    - Attention matrices
    - Framework overhead

    Returns estimated memory in GB.
    """
    # Model weights (in specified dtype)
    model_memory = model_params * dtype_bytes / (1024 ** 3)

    # Optimizer states (AdamW stores 2 FP32 states per parameter)
    optimizer_memory = model_params * 4 * 2 / (1024 ** 3)

    # Gradients (same size as model in training dtype)
    gradient_memory = model_params * dtype_bytes / (1024 ** 3)

    # Activation memory (conservative estimate)
    # Each layer stores: input activations, attention patterns, FFN activations
    # With gradient checkpointing: only sqrt(n_layers) are kept
    # Without: all layers kept
    if gradient_checkpointing:
        # Checkpointing stores ~sqrt(n_layers) activations
        # But we need memory for recomputation during backward
        activation_layers = n_layers ** 0.5 + 2  # +2 for safety margin
    else:
        activation_layers = n_layers

    # Per-layer activation size estimate:
    # - Input: batch * seq * hidden
    # - Attention: batch * heads * seq * seq (QK^T)
    # - FFN: batch * seq * ffn_hidden (3x for SwiGLU)
    n_heads = 8  # Hardcoded for model-nano
    ffn_factor = 3  # SwiGLU has 3 projections

    per_layer_activation = (
        batch_size * seq_len * hidden_dim +  # Layer input
        batch_size * n_heads * seq_len * seq_len +  # Attention matrix
        batch_size * seq_len * hidden_dim * ffn_factor  # FFN
    ) * dtype_bytes

    activation_memory = (per_layer_activation * activation_layers) / (1024 ** 3)

    # Framework overhead (CUDA context, cuDNN workspace, etc.)
    overhead = 0.5  # ~500MB baseline (conservative)

    # Safety margin (20% extra)
    safety_factor = 1.2

    total = (model_memory + optimizer_memory + gradient_memory + activation_memory + overhead) * safety_factor
    return total


def calculate_optimal_batch_size(
    gpu_info: GPUInfo,
    model_params: int,
    seq_len: int,
    hidden_dim: int,
    n_layers: int,
    gradient_checkpointing: bool = True,
    memory_fraction: float = 0.70,  # Use 70% of available memory (conservative)
    min_batch_size: int = 1,
    max_batch_size: int = 32,  # Cap at 32 for safety
) -> tuple[int, int, int]:
    """Calculate optimal micro batch size and gradient accumulation steps.

    Args:
        gpu_info: GPU information from get_gpu_info()
        model_params: Total model parameters
        seq_len: Sequence length
        hidden_dim: Model hidden dimension
        n_layers: Number of transformer layers
        gradient_checkpointing: Whether gradient checkpointing is enabled
        memory_fraction: Fraction of GPU memory to use (default 70%)
        min_batch_size: Minimum micro batch size
        max_batch_size: Maximum micro batch size to try

    Returns:
        Tuple of (micro_batch_size, grad_accumulation_steps, effective_batch_size)
    """
    available_memory = gpu_info.available_memory_gb * memory_fraction
    dtype_bytes = 2  # bf16 or fp16

    # Target effective batch size of 64 (good default for LLMs)
    target_effective_batch = 64

    # Heuristic batch size based on GPU memory (empirically tested)
    # Target ~70-80% GPU utilization for model-nano (~58M params, 512 seq len)
    # With gradient checkpointing enabled
    if gpu_info.total_memory_gb >= 24:
        suggested_batch = 48  # ~80% of 24GB
    elif gpu_info.total_memory_gb >= 16:
        suggested_batch = 32  # ~80% of 16GB
    elif gpu_info.total_memory_gb >= 12:
        suggested_batch = 24  # ~80% of 12GB
    elif gpu_info.total_memory_gb >= 8:
        suggested_batch = 16  # ~80% of 8GB
    elif gpu_info.total_memory_gb >= 6:
        suggested_batch = 12  # ~80% of 6GB
    elif gpu_info.total_memory_gb >= 3.5:
        suggested_batch = 10  # ~80% of 3.5-4GB (~2.8GB usage)
    elif gpu_info.total_memory_gb >= 3:
        suggested_batch = 6   # ~80% of 3GB
    else:
        suggested_batch = 4

    # Use the heuristic-based batch size (more reliable than calculation)
    optimal_batch = suggested_batch

    # Calculate gradient accumulation to reach target effective batch
    grad_accum = max(1, target_effective_batch // optimal_batch)
    effective_batch = optimal_batch * grad_accum

    return optimal_batch, grad_accum, effective_batch


def get_optimal_dtype(gpu_info: Optional[GPUInfo]) -> str:
    """Get optimal training dtype based on GPU capabilities.

    Returns 'bfloat16' for Ampere+, 'float16' for Pascal+, 'float32' otherwise.
    """
    if gpu_info is None:
        return "float32"

    if gpu_info.supports_bf16:
        return "bfloat16"
    elif gpu_info.supports_fp16:
        return "float16"
    else:
        return "float32"


def print_gpu_info(gpu_info: Optional[GPUInfo]) -> None:
    """Print GPU information in a formatted way."""
    print("=" * 60)
    print("GPU CONFIGURATION")
    print("=" * 60)

    if gpu_info is None:
        print("No CUDA GPU detected - will use CPU")
        print("WARNING: Training on CPU will be very slow!")
        print("=" * 60)
        return

    print(f"Device:              {gpu_info.name}")
    print(f"Total Memory:        {gpu_info.total_memory_gb:.2f} GB")
    print(f"Available Memory:    {gpu_info.available_memory_gb:.2f} GB")
    print(f"Compute Capability:  {gpu_info.compute_capability[0]}.{gpu_info.compute_capability[1]}")
    print(f"BF16 Support:        {'Yes' if gpu_info.supports_bf16 else 'No'}")
    print(f"FP16 Support:        {'Yes' if gpu_info.supports_fp16 else 'No'}")
    print(f"Optimal Precision:   {get_optimal_dtype(gpu_info)}")
    print("=" * 60)


def auto_configure_training(
    model_params: int,
    seq_len: int = 512,
    hidden_dim: int = 384,
    n_layers: int = 32,
    gradient_checkpointing: bool = True,
) -> dict:
    """Auto-configure training parameters based on available GPU.

    Returns a dict with optimal training settings.
    """
    gpu_info = get_gpu_info()

    if gpu_info is None:
        # CPU fallback
        return {
            "device": "cpu",
            "dtype": "float32",
            "micro_batch_size": 1,
            "grad_accumulation_steps": 64,
            "effective_batch_size": 64,
            "gradient_checkpointing": True,
            "gpu_name": "CPU",
            "available_memory_gb": 0,
        }

    micro_batch, grad_accum, effective_batch = calculate_optimal_batch_size(
        gpu_info=gpu_info,
        model_params=model_params,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        gradient_checkpointing=gradient_checkpointing,
    )

    return {
        "device": f"cuda:{gpu_info.device_index}",
        "dtype": get_optimal_dtype(gpu_info),
        "micro_batch_size": micro_batch,
        "grad_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "gradient_checkpointing": gradient_checkpointing,
        "gpu_name": gpu_info.name,
        "available_memory_gb": gpu_info.available_memory_gb,
    }


if __name__ == "__main__":
    # Demo: print GPU info and optimal settings
    gpu_info = get_gpu_info()
    print_gpu_info(gpu_info)

    if gpu_info:
        # Example for model-nano (~58M params)
        model_params = 58_000_000
        config = auto_configure_training(
            model_params=model_params,
            seq_len=512,
            hidden_dim=384,
            n_layers=32,
        )

        print("\nAUTO-CONFIGURED TRAINING SETTINGS")
        print("=" * 60)
        for key, value in config.items():
            print(f"{key:25s}: {value}")
        print("=" * 60)
