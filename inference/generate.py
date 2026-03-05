"""Sampling strategies and convenience generation functions."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k sampling: zero out all logits below the top-k values.

    Args:
        logits: (vocab_size,) unnormalized logits for the next token.
        k: Number of top tokens to keep.

    Returns:
        Filtered logits with non-top-k entries set to -inf.
    """
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    top_k_values, _ = torch.topk(logits, k)
    threshold = top_k_values[..., -1]
    logits = logits.clone()
    logits[logits < threshold] = float("-inf")
    return logits


def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling: keep the smallest set of tokens whose
    cumulative probability exceeds p.

    Args:
        logits: (..., vocab_size) unnormalized logits for the next token.
        p: Cumulative probability threshold (0.0 to 1.0).

    Returns:
        Filtered logits with tokens outside the nucleus set to -inf.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Find tokens to remove: those whose cumulative probability exceeds p.
    # Shift right so the first token that pushes past p is kept.
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original ordering
    logits = logits.clone()
    logits.scatter_(-1, sorted_indices, sorted_logits)
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    """Combined sampling: apply temperature, then top-k, then top-p.

    Args:
        logits: (vocab_size,) unnormalized logits for the next token.
        temperature: Sampling temperature. 0.0 means greedy (argmax).
        top_k: If > 0, keep only the top-k tokens before sampling.
        top_p: If < 1.0, apply nucleus sampling after top-k.

    Returns:
        Sampled token ID as a Python int.
    """
    # Greedy decoding
    if temperature <= 0.0:
        return logits.argmax(dim=-1).item()

    # Apply temperature scaling
    logits = logits / temperature

    # Apply top-k filtering
    if top_k > 0:
        logits = sample_top_k(logits, top_k)

    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        logits = sample_top_p(logits, top_p)

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1)
    return token_id.item()


# ---------------------------------------------------------------------------
# Convenience generation functions
# ---------------------------------------------------------------------------

from config import DataConfig
DEFAULT_SYSTEM_PROMPT = DataConfig.system_prompt


def generate_command(engine, query: str) -> str:
    """Generate a git command using greedy (deterministic) decoding.

    Args:
        engine: An InferenceEngine instance.
        query: The user's natural-language query about git.

    Returns:
        The generated git command string.
    """
    prompt = engine.format_prompt(query, system_prompt=DEFAULT_SYSTEM_PROMPT)
    return engine.generate(
        prompt,
        max_new_tokens=256,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        stop_tokens=["<|im_end|>"],
    )


def generate_explanation(engine, query: str) -> str:
    """Generate an explanation with moderate creativity.

    Uses temperature=0.7 and top_p=0.9 for diverse but coherent output.

    Args:
        engine: An InferenceEngine instance.
        query: The user's question about git.

    Returns:
        The generated explanation string.
    """
    prompt = engine.format_prompt(query, system_prompt=DEFAULT_SYSTEM_PROMPT)
    return engine.generate(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_k=0,
        top_p=0.9,
        stop_tokens=["<|im_end|>"],
    )


def generate_completions(engine, partial: str, n: int = 5) -> list[str]:
    """Generate multiple command completions using top-k sampling.

    Args:
        engine: An InferenceEngine instance.
        partial: A partial git command to complete.
        n: Number of completions to generate.

    Returns:
        A list of n completed command strings.
    """
    system = "Complete the following partial git command. Respond with only the completed command."
    prompt = engine.format_prompt(partial, system_prompt=system)
    completions = []
    for _ in range(n):
        result = engine.generate(
            prompt,
            max_new_tokens=128,
            temperature=0.8,
            top_k=10,
            top_p=1.0,
            stop_tokens=["<|im_end|>", "\n"],
        )
        completions.append(result.strip())
    return completions


if __name__ == "__main__":
    # Quick demo of sampling functions with random logits
    print("Sampling strategy demo")
    print("-" * 40)
    dummy_logits = torch.randn(100)

    print(f"Greedy: token {sample_token(dummy_logits, temperature=0.0)}")
    print(f"Temp=1.0: token {sample_token(dummy_logits, temperature=1.0)}")
    print(f"Top-k=10: token {sample_token(dummy_logits, temperature=1.0, top_k=10)}")
    print(f"Top-p=0.9: token {sample_token(dummy_logits, temperature=1.0, top_p=0.9)}")
    print(f"Combined: token {sample_token(dummy_logits, temperature=0.7, top_k=10, top_p=0.9)}")
