"""Global configuration for model-nano: model architecture + training hyperparameters."""

from dataclasses import dataclass, field


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


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

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

    # Batch size
    micro_batch_size: int = 4
    grad_accumulation_steps: int = 16
    # effective batch = micro_batch_size * grad_accumulation_steps = 64

    # Training duration
    max_epochs: int = 20
    max_steps: int = -1  # -1 = use epochs

    # Precision
    dtype: str = "bfloat16"
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


@dataclass
class SFTConfig(TrainingConfig):
    """Supervised fine-tuning specific config."""

    lr: float = 2e-5  # Lower LR for SFT per "Secret Recipe" paper
    min_lr: float = 2e-6
    max_epochs: int = 5
    mask_prompt: bool = True  # Only train on assistant completions


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

    # ChatML template
    system_prompt: str = (
        "You are a Git expert. Provide precise, correct git commands and explanations."
    )
