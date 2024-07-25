import typing
from math import floor
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from segformer_pytorch import Segformer
from torch import nn

# internal
from src.base.base_torch_model import BaseTorchModel


class SegFormerImpl(BaseTorchModel):
    def __init__(
            self,
            config: Dict[str, Any] = None,
            device: torch.device = None
    ):
        super(SegFormerImpl, self).__init__()
        self.config = config
        # [[out_channels_1,Kernelsize_1,ConvSteps_1,Poolingsize_1],...]
        self.network_shape = self.config["network_shape"] if "network_shape" in self.config.keys() and self.config[
            "network_shape"] is not None else [[64, 3, 3, 2], [128, 3, 2, 2], [256, 3, 1, 2]]

        self.num_channels = 1

        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config[
            "verbose"] is not None else False

        self.print_sizes = self.config["print_sizes"] if "print_sizes" in self.config.keys() and self.config[
            "print_sizes"] is not None else False

        self.seg_former = Segformer(
            dims=(32, 64, 160, 256),  # dimensions of each stage
            heads=(1, 2, 5, 8),  # heads of each stage
            ff_expansion=(8, 8, 4, 4),  # feedforward expansion factor of each stage
            reduction_ratio=(8, 4, 2, 1),  # reduction ratio of each stage for efficient attention
            num_layers=4,  # num layers of each stage
            decoder_dim=256,  # decoder dimension
            num_classes=1  # number of segmentation classes
        )

    def forward(self, x):
        res = self.seg_former(x)
        return res
