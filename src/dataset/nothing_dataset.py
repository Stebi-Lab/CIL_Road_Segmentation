from typing import Any, Dict, List, Optional

import numpy as np
import torch
from einops import repeat, rearrange

from src.base.base_torch_dataset import BaseTorchDataset


class TestDataset(BaseTorchDataset):
    def __init__(
            self,
            config: Dict[str, Any],
    ):
        self.config = config
        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config["verbose"] is not None else False
        self.entity_name = self.config["entity_name"] if "entity_name" in self.config.keys() and self.config["entity_name"] is not None else "Test Dataset"
        self.dataset_path = self.config["dataset_path"] if "dataset_path" in self.config.keys() else None
        self.targets = []
        self.data = []
        if self.dataset_path is not None:
            # Here goes data loading & transforms
            pass
        if self.verbose:
            print(f"Loaded dataset {self.entity_name} with {len(self)} samples")
        self.half_precision = self.config["half_precision"] if "half_precision" in self.config.keys() and self.config["half_precision"] is not None else False
        self.device = 'cpu'
        if self.half_precision:
            self.data = [s.astype(np.float16) for s in self.data]
        else:
            self.data = [s.astype(np.float32) for s in self.data]

    def to_device(self, device):
        self.device = device
        self.data = [torch.from_numpy(t) for t in self.data]
        self.targets = [torch.from_numpy(t).long() for t in self.targets]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
