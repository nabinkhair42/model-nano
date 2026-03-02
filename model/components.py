"""Core transformer components: RMSNorm, RoPE, SwiGLU FFN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    10-50% more efficient than LayerNorm with comparable performance.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Precompute RoPE complex frequency tensor.

    Returns shape (max_seq_len, head_dim // 2) complex64 tensor.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # (seq_len, head_dim // 2)
    return torch.polar(torch.ones_like(angles), angles)  # complex64


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """Apply rotary position embeddings to query or key tensor.

    Args:
        x: (batch, seq_len, n_heads, head_dim) tensor
        freqs: precomputed complex frequencies
        start_pos: position offset (for KV-cache inference)
    """
    seq_len = x.shape[1]
    # Reshape x to pairs: (batch, seq, heads, head_dim/2, 2)
    x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    # Convert to complex: (batch, seq, heads, head_dim/2)
    x_complex = torch.view_as_complex(x_pairs)
    # Slice frequencies for this sequence
    freqs_slice = freqs[start_pos : start_pos + seq_len]
    # Broadcast: (1, seq, 1, head_dim/2) * (batch, seq, heads, head_dim/2)
    freqs_slice = freqs_slice.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_slice
    # Convert back to real: (batch, seq, heads, head_dim)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    return x_out.type_as(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network.

    Higher expressivity via gating mechanism. Standard in LLaMA-family models.
    FFN(x) = (Swish(xW1) * xV) W2
    """

    def __init__(self, hidden_dim: int, ffn_hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)  # gate projection
        self.v = nn.Linear(hidden_dim, ffn_hidden_dim, bias=False)   # value projection
        self.w2 = nn.Linear(ffn_hidden_dim, hidden_dim, bias=False)  # down projection
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.v(x)))
