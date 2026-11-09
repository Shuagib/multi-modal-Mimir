from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectorTrainConfig:
    """Stage-1 config: train the projector only, with vision encoder and LM frozen."""

    vision_model_name: str = "google/siglip2-base-patch16-256"
    llm_model_name: str = "danish-foundation-models/DFM-Mimir-v1.5"

    dataset_name: str = "allenai/pixmo-cap"
    dataset_split: str = "train"
    max_samples: int | None = 20_000

    instruction: str = "Describe the image."
    max_caption_tokens: int = 192

    mlp_in_dim: int = 768
    mlp_hidden_dim: int = 2048
    mlp_out_dim: int = 1536
    # Measured std of Mimir's input embedding table. The LM multiplies inputs_embeds
    # by sqrt(hidden_size) internally, so the projector must emit raw-table scale.
    target_rms: float = 0.009060

    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    num_epochs: int = 1

    num_workers: int = 4
    image_cache_dir: Path = PROJECT_ROOT / "data" / "image_cache"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "projector"
    train_dir: Path = PROJECT_ROOT / "data" / "train"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"
    save_every_steps: int = 500
    log_every_steps: int = 10
    seed: int = 42
