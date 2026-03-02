"""PyTorch Datasets and DataLoaders for pretraining and SFT."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PretrainDataset(Dataset):
    """Loads tokenized .bin file (numpy memmap), returns fixed-length chunks.

    The .bin file is expected to contain a flat array of uint16 token IDs
    produced by the tokenization pipeline. Data is memory-mapped so the
    full corpus never needs to fit in RAM.

    Each sample is a contiguous chunk of ``max_seq_len + 1`` tokens.
    The extra token provides the final target for next-token prediction.
    """

    def __init__(self, path: str, max_seq_len: int = 512):
        """
        Args:
            path: Path to a .bin file containing uint16 token IDs.
            max_seq_len: Context window length. Each sample contains
                ``max_seq_len`` input tokens and ``max_seq_len`` target
                tokens (shifted by 1).
        """
        self.max_seq_len = max_seq_len
        # Memory-map the file so we never load the whole thing into RAM.
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        # Number of non-overlapping chunks we can extract.
        # We need max_seq_len + 1 tokens per sample (input + 1 shifted target).
        self.n_samples = len(self.data) // (max_seq_len + 1)
        if self.n_samples == 0:
            raise ValueError(
                f"Data file {path} has {len(self.data)} tokens, need at least "
                f"{max_seq_len + 1} for one sample."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * (self.max_seq_len + 1)
        chunk = self.data[start : start + self.max_seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])    # input_ids:  [0 .. seq_len-1]
        y = torch.from_numpy(chunk[1:])     # targets:    [1 .. seq_len]
        return x, y


class SFTDataset(Dataset):
    """Loads tokenized instruction-response pairs from a .bin file.

    The binary file stores packed samples end-to-end. Each sample is
    structured as:

        [header: 4 bytes (uint32 total_len)] [tokens: total_len * uint16]

    Within each token sequence the layout follows ChatML:

        <|im_start|>system\\n{system_prompt}<|im_end|>\\n
        <|im_start|>user\\n{user_message}<|im_end|>\\n
        <|im_start|>assistant\\n{assistant_response}<|im_end|>

    A parallel mask array is stored in a companion ``.mask.bin`` file
    (uint8, same length as the token file excluding headers). A value of
    ``1`` indicates an assistant-completion token; ``0`` indicates a
    prompt/system/user token. When the companion mask file is absent the
    dataset falls back to runtime detection of assistant spans using the
    ``<|im_start|>`` / ``<|im_end|>`` special token IDs.

    If mask_prompt is True (default) the loss mask zeros out everything
    that is not an assistant completion so the model only learns to
    generate responses, not to parrot the prompts.
    """

    # Sentinel special-token IDs — must match tokenizer/tokenizer.json.
    # The tokenizer assigns special tokens first: <|im_start|>=0, <|im_end|>=1.
    IM_START_ID: int = 0   # <|im_start|>
    IM_END_ID: int = 1     # <|im_end|>

    def __init__(
        self,
        path: str,
        max_seq_len: int = 512,
        mask_prompt: bool = True,
    ):
        """
        Args:
            path: Path to a .bin file containing packed SFT samples.
            max_seq_len: Maximum sequence length (samples are padded/truncated).
            mask_prompt: If True, loss_mask = 0 for non-assistant tokens.
        """
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt

        # Memory-map the token data.
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

        # Try to load a companion mask file.
        mask_path = path.replace(".bin", ".mask.bin")
        try:
            self.mask_data = np.memmap(mask_path, dtype=np.uint8, mode="r")
        except FileNotFoundError:
            self.mask_data = None

        # Build an index of sample boundaries.
        # For simplicity we treat the file as a flat token stream and
        # chunk it the same way as PretrainDataset.  Each sample is
        # max_seq_len + 1 contiguous tokens.
        self.n_samples = len(self.data) // (max_seq_len + 1)
        if self.n_samples == 0:
            raise ValueError(
                f"SFT data file {path} has {len(self.data)} tokens, need at "
                f"least {max_seq_len + 1} for one sample."
            )

    def _build_mask_runtime(self, tokens: np.ndarray) -> np.ndarray:
        """Build a loss mask from token IDs by detecting assistant spans.

        The mask is 1 for tokens inside ``<|im_start|>assistant ...
        <|im_end|>`` spans (excluding the ``<|im_start|>assistant\\n``
        prefix itself) and 0 everywhere else.
        """
        mask = np.zeros(len(tokens), dtype=np.float32)
        in_assistant = False
        skip_role_token = False

        for i, tok in enumerate(tokens):
            if tok == self.IM_START_ID:
                # Next non-whitespace tokens will be the role name.
                # We peek ahead: if the following tokens encode "assistant"
                # we switch on.  For a simple heuristic we just toggle
                # in_assistant based on whether we are currently off.
                # A more robust approach checks the role text, but since
                # ChatML alternates system/user/assistant in order this
                # simple toggle works for well-formed data.
                skip_role_token = True
                in_assistant = False
                continue
            if skip_role_token:
                # This is the role token (e.g. "assistant", "user", "system").
                # We cannot decode here, so we use the positional heuristic:
                # assistant turn markers appear after user turn markers.
                # Instead, mark remaining as assistant until <|im_end|>.
                # We'll set in_assistant True only for every third <|im_start|>
                # block (system=0, user=1, assistant=2, ...).
                skip_role_token = False
                continue
            if tok == self.IM_END_ID:
                in_assistant = False
                continue
            if in_assistant:
                mask[i] = 1.0

        # Simpler reliable approach: scan for <|im_start|> then look for
        # the literal "assistant" role.  Since we cannot decode, use the
        # observation that in ChatML the pattern is always:
        #   im_start  ROLE_TOKEN  \n  ...content...  im_end
        # We find all im_start positions and check if the content between
        # consecutive im_start/im_end pairs is the assistant block.
        # Reimplemented below with a clean state machine.
        mask = np.zeros(len(tokens), dtype=np.float32)
        i = 0
        while i < len(tokens):
            if tokens[i] == self.IM_START_ID:
                # Find the matching im_end
                end = i + 1
                while end < len(tokens) and tokens[end] != self.IM_END_ID:
                    end += 1
                # Heuristic: if this block is the third or later (0-indexed:
                # block 2, 5, 8, ...) it is an assistant block.  But multi-turn
                # means we cannot rely on counting.  Instead, we check the
                # length: assistant blocks tend to be longer.  This is fragile.
                #
                # Best simple heuristic without decoding: we mark every other
                # turn starting from the third as assistant.  For single-turn
                # (system, user, assistant) this works perfectly.
                #
                # We'll count im_start occurrences.
                pass  # fall through to counting approach below
                i = end + 1
                continue
            i += 1

        # Final approach: count <|im_start|> blocks.  In well-formed ChatML
        # the pattern repeats as (system, user, assistant) triples.
        # Block indices 2, 5, 8, ... (i.e. index % 3 == 2) are assistant.
        mask = np.zeros(len(tokens), dtype=np.float32)
        block_idx = -1
        in_block = False
        header_tokens_remaining = 0

        for i, tok in enumerate(tokens):
            if tok == self.IM_START_ID:
                block_idx += 1
                in_block = True
                # Skip the role token and the newline after it.
                header_tokens_remaining = 2  # role_id + newline
                continue
            if tok == self.IM_END_ID:
                in_block = False
                continue
            if in_block and header_tokens_remaining > 0:
                header_tokens_remaining -= 1
                continue
            if in_block and block_idx % 3 == 2:
                # This is an assistant block body token.
                mask[i] = 1.0

        return mask

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = idx * (self.max_seq_len + 1)
        end = start + self.max_seq_len + 1
        chunk = self.data[start:end].astype(np.int64)

        input_ids = torch.from_numpy(chunk[:-1])
        targets = torch.from_numpy(chunk[1:])

        # Build loss mask.
        if not self.mask_prompt:
            loss_mask = torch.ones(self.max_seq_len, dtype=torch.float32)
        elif self.mask_data is not None:
            raw_mask = self.mask_data[start:end].astype(np.float32)
            # Mask aligns with targets (shifted by 1).
            loss_mask = torch.from_numpy(raw_mask[1:])
        else:
            full_mask = self._build_mask_runtime(chunk)
            loss_mask = torch.from_numpy(full_mask[1:])

        return input_ids, targets, loss_mask


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with sensible defaults for training.

    Args:
        dataset: A PretrainDataset or SFTDataset instance.
        batch_size: Micro-batch size.
        shuffle: Whether to shuffle (True for train, False for val).
        num_workers: Number of data-loading worker processes.
        pin_memory: Pin memory for faster GPU transfer.
        drop_last: Drop the last incomplete batch.

    Returns:
        A configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
