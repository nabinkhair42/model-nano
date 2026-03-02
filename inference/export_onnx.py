"""ONNX export and INT8 quantization for NanoGPT.

This module exports the NanoGPT model to ONNX format and optionally applies
INT8 dynamic quantization via onnxruntime for efficient CPU inference.

NOTE: KV-cache is NOT included in the ONNX export for simplicity. The exported
model expects full input sequences each time (no incremental decoding). For
production use with KV-cache, you would need to export the cache tensors as
additional inputs/outputs, which significantly complicates the ONNX graph
and is model-serving-framework-specific.
"""

import sys
from pathlib import Path

import torch

# Ensure the project root is importable when running this file directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import ModelConfig
from model.transformer import NanoGPT


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
) -> tuple[NanoGPT, ModelConfig]:
    """Load a NanoGPT model from a training checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model onto.

    Returns:
        A tuple of (model, config).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint and isinstance(checkpoint["config"], dict):
        config = ModelConfig(**checkpoint["config"])
    elif "config" in checkpoint and isinstance(checkpoint["config"], ModelConfig):
        config = checkpoint["config"]
    else:
        config = ModelConfig()

    model = NanoGPT(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


class NanoGPTONNXWrapper(torch.nn.Module):
    """Thin wrapper around NanoGPT that simplifies the forward signature for ONNX export.

    ONNX export works best with a simple (input -> output) signature.
    This wrapper:
      - Drops targets (not needed at inference time).
      - Drops KV-cache inputs/outputs (simplifies the export; see module docstring).
      - Returns only the logits tensor.
    """

    def __init__(self, model: NanoGPT):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return logits only.

        Args:
            input_ids: (batch, seq_len) integer token IDs.

        Returns:
            logits: (batch, seq_len, vocab_size) float tensor.
        """
        logits, _, _ = self.model(input_ids, targets=None, start_pos=0, kv_caches=None)
        return logits


def export_to_onnx(
    model: NanoGPT,
    config: ModelConfig,
    output_path: str,
    opset_version: int = 17,
) -> str:
    """Export the model to ONNX format.

    Args:
        model: A NanoGPT model instance (already in eval mode).
        config: The ModelConfig used to build the model.
        output_path: Destination path for the .onnx file.
        opset_version: ONNX opset version (17 is widely supported and recent).

    Returns:
        The resolved output path string.
    """
    wrapper = NanoGPTONNXWrapper(model)
    wrapper.eval()

    # Create a dummy input matching the model's expected input shape.
    # Use a moderate sequence length for the export trace.
    dummy_input = torch.randint(
        0, config.vocab_size, (1, config.max_seq_len), dtype=torch.long
    )

    output_path = str(Path(output_path).resolve())

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"ONNX model exported to: {output_path}")
    return output_path


def quantize_int8(
    onnx_model_path: str,
    output_path: str | None = None,
) -> str:
    """Apply INT8 dynamic quantization to an ONNX model using onnxruntime.

    Dynamic quantization quantizes weights to INT8 at export time and
    dynamically quantizes activations at runtime. This reduces model size
    by ~4x and can speed up CPU inference with minimal accuracy loss.

    Args:
        onnx_model_path: Path to the input .onnx model file.
        output_path: Path for the quantized .onnx output. If None, appends
            ``_int8`` to the original filename.

    Returns:
        The resolved output path of the quantized model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError(
            "onnxruntime is required for INT8 quantization. "
            "Install it with: pip install onnxruntime"
        )

    onnx_model_path = str(Path(onnx_model_path).resolve())

    if output_path is None:
        p = Path(onnx_model_path)
        output_path = str(p.with_stem(p.stem + "_int8"))

    output_path = str(Path(output_path).resolve())

    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )

    original_size = Path(onnx_model_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)

    print(f"Quantized model saved to: {output_path}")
    print(f"Original size:  {original_size:.1f} MB")
    print(f"Quantized size: {quantized_size:.1f} MB")
    print(f"Compression:    {original_size / quantized_size:.2f}x")

    return output_path


def validate_onnx(onnx_model_path: str, config: ModelConfig) -> bool:
    """Validate the exported ONNX model by running a test inference.

    Args:
        onnx_model_path: Path to the .onnx file.
        config: Model config for constructing a test input.

    Returns:
        True if validation passes.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping validation.")
        return False

    try:
        import onnx
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structure validation passed.")
    except ImportError:
        print("onnx package not installed, skipping structural check.")

    session = ort.InferenceSession(onnx_model_path)

    # Run a test inference
    import numpy as np

    test_input = np.random.randint(0, config.vocab_size, (1, 16)).astype(np.int64)
    outputs = session.run(None, {"input_ids": test_input})

    logits = outputs[0]
    expected_shape = (1, 16, config.vocab_size)
    assert logits.shape == expected_shape, (
        f"Output shape mismatch: got {logits.shape}, expected {expected_shape}"
    )

    print(f"Inference validation passed. Output shape: {logits.shape}")
    return True


def main():
    """CLI entry point for ONNX export and quantization."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export NanoGPT to ONNX and optionally quantize to INT8.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m inference.export_onnx --checkpoint checkpoints/model.pt\n"
            "  python -m inference.export_onnx --checkpoint checkpoints/model.pt --quantize\n"
            "  python -m inference.export_onnx --checkpoint checkpoints/model.pt "
            "--output model.onnx --quantize --validate\n"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_nano.onnx",
        help="Output path for the ONNX model (default: model_nano.onnx).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply INT8 dynamic quantization after export.",
    )
    parser.add_argument(
        "--quantized-output",
        type=str,
        default=None,
        help="Output path for the quantized model (default: adds _int8 suffix).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on the exported model.",
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, device="cpu")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count:,} parameters")

    # Export to ONNX
    print(f"\nExporting to ONNX (opset {args.opset})...")
    onnx_path = export_to_onnx(model, config, args.output, opset_version=args.opset)

    # Validate
    if args.validate:
        print("\nValidating ONNX model...")
        validate_onnx(onnx_path, config)

    # Quantize
    if args.quantize:
        print("\nApplying INT8 dynamic quantization...")
        quantized_path = quantize_int8(onnx_path, args.quantized_output)

        if args.validate:
            print("\nValidating quantized model...")
            validate_onnx(quantized_path, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
