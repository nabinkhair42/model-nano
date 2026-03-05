"""Global configuration for model-nano: model architecture + training hyperparameters.

Supports auto-detection of GPU capabilities for optimal training settings.
Set micro_batch_size=-1 to enable auto-detection.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Transformer architecture configuration."""

    # Vocabulary & embeddings
    vocab_size: int = 16_384
    max_seq_len: int = 512

    # Transformer dimensions
    n_layers: int = 32
    hidden_dim: int = 384
    ffn_hidden_dim: int = 1_024  # 2.67x hidden, SwiGLU gated

    # Attention
    n_heads: int = 8
    n_kv_heads: int = 4  # GQA: 2 query heads per KV head
    head_dim: int = 48  # hidden_dim // n_heads

    # RoPE
    rope_theta: float = 10_000.0

    # RMSNorm
    norm_eps: float = 1e-6

    # Dropout (0.0 for small models with enough data)
    dropout: float = 0.0

    def __post_init__(self):
        assert self.hidden_dim % self.n_heads == 0, "hidden_dim must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = self.hidden_dim // self.n_heads

    def count_parameters(self) -> int:
        """Estimate total model parameters."""
        # Embeddings
        embedding_params = self.vocab_size * self.hidden_dim

        # Per-layer parameters
        # Attention: Q, K, V projections + output projection
        attn_params = (
            self.hidden_dim * self.hidden_dim +  # Q
            self.hidden_dim * (self.hidden_dim // (self.n_heads // self.n_kv_heads)) +  # K (GQA)
            self.hidden_dim * (self.hidden_dim // (self.n_heads // self.n_kv_heads)) +  # V (GQA)
            self.hidden_dim * self.hidden_dim    # Output
        )
        # FFN: SwiGLU has 3 projections (gate, up, down)
        ffn_params = 3 * self.hidden_dim * self.ffn_hidden_dim
        # Norms: 2 per layer (attention + ffn)
        norm_params = 2 * self.hidden_dim

        per_layer = attn_params + ffn_params + norm_params
        transformer_params = self.n_layers * per_layer

        # Final norm + LM head (tied with embeddings, so not counted twice)
        final_norm = self.hidden_dim

        return embedding_params + transformer_params + final_norm


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Set micro_batch_size=-1 to auto-detect optimal batch size based on GPU memory.
    Set dtype="auto" to auto-detect optimal precision (bf16/fp16/fp32).
    """

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    min_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1

    # LR schedule
    warmup_ratio: float = 0.02
    lr_schedule: str = "cosine"

    # Batch size (-1 = auto-detect based on GPU memory)
    micro_batch_size: int = -1
    grad_accumulation_steps: int = -1  # -1 = auto-calculate to reach effective_batch
    target_effective_batch: int = 64   # Target effective batch size
    # effective batch = micro_batch_size * grad_accumulation_steps

    # Training duration
    max_epochs: int = 20
    max_steps: int = -1  # -1 = use epochs

    # Precision ("auto" = detect based on GPU capability)
    dtype: str = "auto"
    gradient_checkpointing: bool = True

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Logging & checkpointing
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"

    # Wandb
    wandb_project: str = "model-nano"
    wandb_run: str = ""
    use_wandb: bool = False

    def auto_configure(self, model_config: 'ModelConfig') -> 'TrainingConfig':
        """Auto-configure batch size and dtype based on GPU capabilities.

        Returns a new TrainingConfig with optimal settings.
        """
        from training.gpu_utils import auto_configure_training, get_gpu_info, print_gpu_info

        # Print GPU info
        gpu_info = get_gpu_info()
        print_gpu_info(gpu_info)

        # Get model param count
        model_params = model_config.count_parameters()

        # Auto-configure
        auto_config = auto_configure_training(
            model_params=model_params,
            seq_len=model_config.max_seq_len,
            hidden_dim=model_config.hidden_dim,
            n_layers=model_config.n_layers,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        # Update config with auto-detected values
        if self.micro_batch_size == -1:
            self.micro_batch_size = auto_config["micro_batch_size"]

        if self.grad_accumulation_steps == -1:
            self.grad_accumulation_steps = auto_config["grad_accumulation_steps"]

        if self.dtype == "auto":
            self.dtype = auto_config["dtype"]

        print(f"\nAUTO-CONFIGURED TRAINING:")
        print(f"  Micro batch size:     {self.micro_batch_size}")
        print(f"  Grad accumulation:    {self.grad_accumulation_steps}")
        print(f"  Effective batch:      {self.micro_batch_size * self.grad_accumulation_steps}")
        print(f"  Precision:            {self.dtype}")
        print(f"  Gradient checkpoint:  {self.gradient_checkpointing}")

        return self


@dataclass
class SFTConfig(TrainingConfig):
    """Supervised fine-tuning specific config.

    Inherits auto-configuration from TrainingConfig.
    Uses lower learning rate per "Secret Recipe" paper recommendations.
    """

    lr: float = 2e-5  # Lower LR for SFT per "Secret Recipe" paper
    min_lr: float = 2e-6
    max_epochs: int = 3  # Reduced to prevent overfitting (was 5)
    mask_prompt: bool = True  # Only train on assistant completions

    # Early stopping to prevent overfitting
    early_stopping_patience: int = 3  # Stop if val_loss doesn't improve for N evals


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    # Paths
    raw_dir: str = "data/raw"
    processed_dir: str = "data"
    tokenizer_path: str = "tokenizer/tokenizer.json"

    # Tokenizer
    vocab_size: int = 16_384
    min_frequency: int = 2

    # Dataset
    max_seq_len: int = 512
    train_split: float = 0.95
    val_split: float = 0.05
    seed: int = 42

    # Special tokens
    special_tokens: list = field(default_factory=lambda: [
        "<|im_start|>",
        "<|im_end|>",
        "<|pad|>",
    ])

    # ChatML template - comprehensive system prompt for precise outputs
    system_prompt: str = (
        "You are a Git and GitHub expert assistant. Your role is to help developers with:\n"
        "1. Git commands: Provide exact, copy-paste ready commands\n"
        "2. Explanations: Clear, concise explanations of git concepts\n"
        "3. Error diagnosis: Identify issues and provide step-by-step fixes\n\n"
        "Guidelines:\n"
        "- Always provide the exact command first, then explain\n"
        "- Use code blocks for commands: ```git command```\n"
        "- Be concise but complete\n"
        "- Warn about destructive operations (reset --hard, force push)"
    )
