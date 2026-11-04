import json, logging, math, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from language_model.model import Mimir
from projector.model import MLP
from pipeline.training.config import ProjectorTrainConfig

NUM_PATCHES = 256

class FeatureDataset(Dataset):
    """Precomputed SigLIP2 feature + captions from extract_vm_features.py."""