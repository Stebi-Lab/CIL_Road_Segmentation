import abc
from typing import Any, Dict

import torch
from torch.utils.data import Dataset


class BaseTorchDataset(Dataset, metaclass=abc.ABCMeta):

    _config_group_ = "trainer/dataset"
    _config_name_ = "default"

    def to_device(self, device: torch.device) -> None:
        pass
