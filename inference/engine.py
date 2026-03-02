"""KV-cache inference engine for NanoGPT."""

import sys
from pathlib import Path

import torch

# Ensure the project root is importable when running this file directly.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import ModelConfig
from model.transformer import NanoGPT
from inference.generate import sample_token

# Try to import the tokenizers library
from tokenizers import Tokenizer


DEFAULT_SYSTEM_PROMPT = (
    "You are a Git expert. Provide precise, correct git commands and explanations."
)


class InferenceEngine:
    """KV-cache inference engine for NanoGPT.

    Loads a trained model checkpoint and tokenizer, then provides efficient
    autoregressive text generation with KV-cache reuse.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str = "auto",
    ):
        """Load model and tokenizer, set up for inference.

        Args:
            model_path: Path to a saved model checkpoint (.pt file).
                The checkpoint should contain at least a ``model_state_dict``
                key. It may optionally contain a ``config`` key with a
                ModelConfig dict for architecture reconstruction.
            tokenizer_path: Path to tokenizer.json (Hugging Face tokenizers format).
            device: Target device. ``"auto"`` selects CUDA when available,
                otherwise falls back to CPU.
        """
        # ------------------------------------------------------------------
        # Device selection
        # ------------------------------------------------------------------
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ------------------------------------------------------------------
        # Load tokenizer
        # ------------------------------------------------------------------
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Cache special-token IDs for fast lookup
        self._im_start_id = self.tokenizer.token_to_id("<|im_start|>")
        self._im_end_id = self.tokenizer.token_to_id("<|im_end|>")
        self._pad_id = self.tokenizer.token_to_id("<|pad|>")

        # ------------------------------------------------------------------
        # Load model
        # ------------------------------------------------------------------
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Reconstruct config from checkpoint if available, otherwise use defaults.
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            config = ModelConfig(**checkpoint["config"])
        elif "config" in checkpoint and isinstance(checkpoint["config"], ModelConfig):
            config = checkpoint["config"]
        else:
            config = ModelConfig()

        self.config = config
        self.model = NanoGPT(config)

        # Handle both "model_state_dict" and bare state-dict checkpoints.
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Tokenize text into a list of token IDs.

        Args:
            text: Raw input string.

        Returns:
            List of integer token IDs.
        """
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token IDs back into a string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        return self.tokenizer.decode(token_ids)

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, query: str, system_prompt: str | None = None) -> str:
        """Format a user query into a ChatML conversation template.

        The produced format is:

            <|im_start|>system
            {system_prompt}<|im_end|>
            <|im_start|>user
            {query}<|im_end|>
            <|im_start|>assistant

        Args:
            query: The user's message / question.
            system_prompt: Optional system-level instruction. Defaults to the
                standard git-expert system prompt.

        Returns:
            Fully formatted prompt string ready for tokenization.
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_tokens: list[str] | None = None,
    ) -> str:
        """Generate text with KV-cache for efficiency.

        Performs a two-phase generation:
          1. **Prefill** -- Run the entire prompt through the model in one
             forward pass to produce initial KV caches and the first set of
             logits.
          2. **Decode** -- Autoregressively generate one token at a time,
             feeding only the newly generated token and reusing the cached
             key/value tensors from all previous positions.

        Args:
            prompt: The input text (already formatted if needed).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature. 0.0 gives greedy (argmax).
            top_k: If > 0, restrict sampling to the top-k tokens.
            top_p: If < 1.0, apply nucleus sampling.
            stop_tokens: Optional list of strings that, when generated,
                signal the end of generation (the stop token itself is not
                included in the output).

        Returns:
            The generated text (excluding the original prompt).
        """
        # Resolve stop-token IDs
        stop_ids: set[int] = set()
        if stop_tokens:
            for tok_str in stop_tokens:
                tok_id = self.tokenizer.token_to_id(tok_str)
                if tok_id is not None:
                    stop_ids.add(tok_id)

        # Encode prompt
        input_ids = self.encode(prompt)

        # Truncate to max_seq_len to prevent out-of-bounds RoPE access
        max_prompt_len = self.config.max_seq_len - 1  # leave room for at least 1 new token
        if len(input_ids) > max_prompt_len:
            input_ids = input_ids[-max_prompt_len:]

        prompt_len = len(input_ids)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        # ------------------------------------------------------------------
        # Phase 1: Prefill -- run full prompt through model
        # ------------------------------------------------------------------
        logits, _, kv_caches = self.model(input_tensor, targets=None, start_pos=0, kv_caches=None)
        # logits shape: (1, prompt_len, vocab_size)
        # Take logits for the last prompt token to predict the first new token
        next_logits = logits[0, -1, :]  # (vocab_size,)

        # ------------------------------------------------------------------
        # Phase 2: Autoregressive decode with KV-cache
        # ------------------------------------------------------------------
        generated_ids: list[int] = []

        for step in range(max_new_tokens):
            # Sample the next token
            next_token = sample_token(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

            # Check stop conditions
            if next_token in stop_ids:
                break

            generated_ids.append(next_token)

            # Check if we hit the sequence length limit
            current_pos = prompt_len + len(generated_ids)
            if current_pos >= self.config.max_seq_len:
                break

            # Prepare input for next step: just the new token
            next_input = torch.tensor(
                [[next_token]], dtype=torch.long, device=self.device
            )  # (1, 1)

            # Forward pass with KV-cache. start_pos = total tokens seen so far - 1
            # because we are providing exactly 1 new token at position (current_pos - 1).
            logits, _, kv_caches = self.model(
                next_input,
                targets=None,
                start_pos=current_pos - 1,
                kv_caches=kv_caches,
            )
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Check for multi-token stop strings in the decoded output.
            # Single-token stop strings are already handled by stop_ids above,
            # but some stop strings may span multiple tokens.
            if stop_tokens:
                generated_text = self.decode(generated_ids)
                for stop_str in stop_tokens:
                    if stop_str in generated_text:
                        # Remove the stop string and anything after it
                        return generated_text.split(stop_str)[0]

        return self.decode(generated_ids)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NanoGPT Inference Engine")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Path to tokenizer.json",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--query", type=str, required=True, help="Query to send to the model")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")

    args = parser.parse_args()

    engine = InferenceEngine(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
    )

    prompt = engine.format_prompt(args.query)
    print(f"Prompt:\n{prompt}\n")
    print("Generating...")
    output = engine.generate(
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_tokens=["<|im_end|>"],
    )
    print(f"\nResponse:\n{output}")
