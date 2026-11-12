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

    mlp_in_dim: int = 768
    mlp_hidden_dim: int = 2048
    mlp_out_dim: int = 1536
    # Measured std of Mimir's input embedding table. The LM multiplies inputs_embeds
    # by sqrt(hidden_size) internally, so the projector must emit raw-table scale.
    target_rms: float = 0.009060
    epochs = 10

    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 20

    image_cache_dir: Path = PROJECT_ROOT / "data" / "image_cache"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "projector"
    train_dir: Path = PROJECT_ROOT / "data" / "train"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"
