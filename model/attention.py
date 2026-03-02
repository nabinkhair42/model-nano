"""Grouped-Query Attention with KV-cache support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components import apply_rope


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Causal Self-Attention.

    Uses fewer KV heads than query heads to save memory
    with negligible quality loss at this scale.

    Args:
        hidden_dim: Model hidden dimension
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads (must divide n_heads)
        head_dim: Dimension per head
        dropout: Attention dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads  # how many Q heads per KV head

        self.wq = nn.Linear(hidden_dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(hidden_dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            rope_freqs: precomputed RoPE frequencies
            mask: causal attention mask
            start_pos: position offset for KV-cache
            kv_cache: optional (cached_k, cached_v) from previous steps

        Returns:
            output: (batch, seq_len, hidden_dim)
            new_kv_cache: (keys, values) for caching
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.wq(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Apply RoPE to Q and K
        q = apply_rope(q, rope_freqs, start_pos)
        k = apply_rope(k, rope_freqs, start_pos)

        # KV-cache: append new K, V to cached
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=1)
            v = torch.cat([cached_v, v], dim=1)

        new_kv_cache = (k, v)

        # Expand KV heads to match Q heads (GQA)
        if self.n_rep > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            k = k.reshape(batch, -1, self.n_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, self.n_rep, -1)
            v = v.reshape(batch, -1, self.n_heads, self.head_dim)

        # Transpose to (batch, heads, seq, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, v)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden_dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.wo(output), new_kv_cache
