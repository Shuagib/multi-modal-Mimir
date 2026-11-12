import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """Precomputed SigLIP2 feature + captions from extract_vm_features.py."""
    def __init__(self, shard_dir: Path) -> None:
        self.shard_dir = Path(shard_dir)
        self.feature_files = sorted(self.shard_dir.glob("features-*.npy"))
        if not self.feature_files:
            raise FileNotFoundError(f"no feature shards in {shard_dir}")

        self.captions: list[list[str]] = []
        self.offsets: list[int] = []
        self.total = 0 
        self.init_feature_files()
        self._open: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict:
        """Returns {features, captions}"""
        shard = int(np.searchsorted(self.offsets, index, side="right")) - 1 # returns the position where index would be inserted to keep the list sorted.
        local = index - self.offsets[shard]
        if shard not in self._open:
            self._open[shard] = np.load(self.feature_files[shard], mmap_mode="r")
        return {
            "features": torch.from_numpy(np.array(self._open[shard][local])),
            "caption": self.captions[shard][local]
        }

    def init_feature_files(self) -> list:
        for path in self.feature_files:
            jsonl = path.parent / path.name.replace("features-", "caption-").replace(".npy", ".jsonl")
            captions = [json.loads(line)["caption"] for line in jsonl.open()]
            features_len = np.load(path, mmap_mode="r").shape[0]
            if features_len != len(captions):
                raise ValueError(f"{path.name}")
            self.captions.append(captions)
            self.offsets.append(self.total)
            self.total =+ len(captions)