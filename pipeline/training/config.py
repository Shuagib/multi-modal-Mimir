from dataclasses import dataclass
from pathlib import Path
import os

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectorTrainConfig:
    """Stage-1 config: train the projector only, with vision encoder and Mimir ."""

    vision_model_name: str = "google/siglip2-base-patch16-256"
    llm_model_name: str = "danish-foundation-models/DFM-Mimir-v1.5"

    dataset_name: str = "allenai/pixmo-cap"
    dataset_split: str = "train"
    max_samples: int | None = 20_000

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    instruction: str = "Describe the image."

    mlp_in_dim: int = 768
    mlp_hidden_dim: int = 2048
    mlp_out_dim: int = 1536
  
    target_rms: float = 0.010334 # by language_model/analysis/rms.py

    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = int(os.environ.get("EPOCHS", 5))

    image_cache_dir: Path = PROJECT_ROOT / "data" / "image_cache"
    log_dir: Path = PROJECT_ROOT / "pipeline" / "training" / "logs"
    checkpoint_dir: Path = PROJECT_ROOT / "pipeline" / "training" / "model_saves"

    # train/eval/test dir
    TRAIN_COUNT = os.environ.get("TRAIN_COUNT", 
        #"1k"
        #"1k-git"
        "10k"
        #"28k"
        #"76k"
    )

    train_dir: Path = PROJECT_ROOT / f"data-{TRAIN_COUNT}" / "train"
    eval_dir:  Path = PROJECT_ROOT / f"data-{TRAIN_COUNT}" / "eval"
    test_dir:  Path = PROJECT_ROOT / f"data-{TRAIN_COUNT}" / "test"
    


@dataclass(frozen=True)
class ProjectorTrainConfigHRM:
    """Stage-1 config: train the projector only, with vision encxcoder and HRM Model."""

    vision_model_name: str = "google/siglip2-base-patch16-256"
    llm_model_name: str = "sapientinc/HRM-Text-1B"

    dataset_name: str = "allenai/pixmo-cap"
    dataset_split: str = "train"
    max_samples: int | None = 20_000

    instruction: str = "Describe the image."

    mlp_in_dim: int = 768
    mlp_hidden_dim: int = 2048
    mlp_out_dim: int = 1536
  
    target_rms: float = 0.009060
    epochs = 1

    batch_size: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 20

    image_cache_dir: Path = PROJECT_ROOT / "data" / "image_cache"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "projector_hrm"
    train_dir: Path = PROJECT_ROOT / "data" / "train"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"



@dataclass(frozen=True)
class ProjectorTrainConfigQwen:
    """Stage-1 config: train the projector only, with vision encoder and Qwen"""

    vision_model_name: str = "google/siglip2-base-patch16-256"
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"

    dataset_name: str = "allenai/pixmo-cap"
    dataset_split: str = "train"
    max_samples: int | None = 20_000

    instruction: str = "Describe the image."

    mlp_in_dim: int = 768
    mlp_hidden_dim: int = 2048
    mlp_out_dim: int = 1536
  
    target_rms: float = 0.026090
    epochs = 1

    batch_size: int = 2 
    learning_rate: float = 1e-3
    num_epochs: int = 20

    image_cache_dir: Path = PROJECT_ROOT / "data" / "image_cache"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints" / "projector_qwen"
    report_dir: Path = PROJECT_ROOT / "report"
    train_dir: Path = PROJECT_ROOT / "data" / "train"
    eval_dir: Path = PROJECT_ROOT / "data" / "eval"

