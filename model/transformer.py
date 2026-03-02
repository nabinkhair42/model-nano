"""Full NanoGPT transformer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from model.components import RMSNorm, SwiGLUFFN, precompute_rope_frequencies
from model.attention import GroupedQueryAttention


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.attn = GroupedQueryAttention(
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            dropout=config.dropout,
        )
        self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.ffn = SwiGLUFFN(
            hidden_dim=config.hidden_dim,
            ffn_hidden_dim=config.ffn_hidden_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Pre-norm attention with residual
        h, new_kv_cache = self.attn(
            self.attn_norm(x), rope_freqs, mask, start_pos, kv_cache
        )
        x = x + h

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_kv_cache


class NanoGPT(nn.Module):
    """NanoGPT: ~45M parameter transformer for git command assistance.

    Architecture:
        - 32 layers, 384 hidden dim, 8 heads (4 KV heads GQA)
        - RMSNorm (pre-norm), RoPE, SwiGLU FFN
        - 16,384 vocab, 512 max sequence length
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()

        # Token embeddings (no positional — using RoPE)
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.hidden_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(self.config) for _ in range(self.config.n_layers)
        ])

        # Final norm + output projection
        self.norm = RMSNorm(self.config.hidden_dim, self.config.norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_dim, self.config.vocab_size, bias=False)

        # Weight tying: share embedding weights with output projection
        self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE frequencies (registered as buffer, not parameter)
        rope_freqs = precompute_rope_frequencies(
            self.config.head_dim,
            self.config.max_seq_len,
            self.config.rope_theta,
        )
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections per GPT-2 paper
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * self.config.n_layers) ** 0.5)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            targets: (batch, seq_len) target token IDs for loss computation
            start_pos: position offset for KV-cache inference
            kv_caches: list of (k, v) caches per layer

        Returns:
            logits: (batch, seq_len, vocab_size)
            loss: scalar loss if targets provided, else None
            new_kv_caches: updated KV caches for each layer
        """
        batch, seq_len = input_ids.shape

        # Token embeddings
        x = self.tok_emb(input_ids)

        # Build causal mask (only needed during training / prefill)
        mask = None
        if seq_len > 1:
            mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype
            )
            mask = torch.triu(mask, diagonal=1)
            if start_pos > 0:
                # During cached inference prefill, extend mask for cached positions
                mask = torch.cat(
                    [torch.zeros(seq_len, start_pos, device=x.device, dtype=x.dtype), mask],
                    dim=-1,
                )

        # Run through transformer blocks
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, self.rope_freqs, mask, start_pos, cache)
            new_kv_caches.append(new_cache)

        # Final norm + project to vocab
        x = self.norm(x)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_kv_caches

    def count_parameters(self) -> dict[str, int]:
        """Return parameter count breakdown."""
        embedding = sum(p.numel() for n, p in self.named_parameters() if "tok_emb" in n)
        total = sum(p.numel() for p in self.parameters())
        # lm_head is tied, so don't double-count
        return {
            "embedding": embedding,
            "transformer": total - embedding,
            "total": total,
        }
