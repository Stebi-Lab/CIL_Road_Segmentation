import typing
from math import floor
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# internal
from src.base.base_torch_model import BaseTorchModel


class CombiModelMLP(BaseTorchModel):
    def __init__(
            self,
            config: Dict[str, Any] = None,
            device: torch.device = None
    ):
        super(CombiModelMLP, self).__init__()
        self.config = config

        self.padding = self.config["padding"] if "padding" in self.config.keys() and self.config[
            "padding"] is not None else [0, 0, 0, 0]

        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config[
            "verbose"] is not None else False

        self.print_sizes = self.config["print_sizes"] if "print_sizes" in self.config.keys() and self.config[
            "print_sizes"] is not None else False
        if "part1_checkpoint_path" not in self.config.keys() or self.config["part1_checkpoint_path"] is None:
            raise ValueError("No part 1 path")
        self.p1_checkpoint_path = self.config["part1_checkpoint_path"]
        self.p1_config = self.config["part1_config"] if "part1_config" in self.config.keys() and self.config[
            "part1_config"] is not None else {}

        if "part2_checkpoint_path" not in self.config.keys() or self.config["part2_checkpoint_path"] is None:
            raise ValueError("No part 2 path")
        self.p2_checkpoint_path = self.config["part2_checkpoint_path"]
        self.p2_config = self.config["part2_config"] if "part2_config" in self.config.keys() and self.config[
            "part2_config"] is not None else {}

        self.device = device

        from .seg_former import SegFormerImpl
        from .unet_plusplus_pretrained import UNetPlusPlusModel_Pretrained

        self.p1 = SegFormerImpl(self.p1_config)
        checkpoint1 = torch.load(self.p1_checkpoint_path, map_location=self.device)
        self.p1.load_state_dict(checkpoint1["model"])
        # self.p1.to(self.device)
        for param in self.p1.parameters():
            param.requires_grad = False

        self.p2 = UNetPlusPlusModel_Pretrained(self.p2_config)
        checkpoint2 = torch.load(self.p2_checkpoint_path, map_location=self.device)
        self.p2.load_state_dict(checkpoint2["model"])
        # self.p2.to(self.device)
        for param in self.p2.parameters():
            param.requires_grad = False

        self.dimx = 400 + self.padding[2]
        self.dimy = 400 + self.padding[3]

        self.mlp = nn.Sequential(
            nn.Linear(2 * 1 * self.dimx * self.dimy, 1024),  # Adjust input dimension based on flattened size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1 * self.dimx * self.dimy)  # Adjust output dimension as needed
        )

    def unfreeze_weights(self):
        for param in self.p1.parameters():
            param.requires_grad = True
        for param in self.p2.parameters():
            param.requires_grad = True

    def forward(self, x):
        out1 = self.p1(x)
        out1 = F.interpolate(out1, size=(self.dimx, self.dimy), mode='bilinear', align_corners=False)

        out2 = self.p2(x)

        out1_flat = out1.view(out1.size(0), -1)
        out2_flat = out2.view(out2.size(0), -1)

        combined = torch.cat((out1_flat, out2_flat), dim=1)

        fused = self.mlp(combined)
        fused = fused.view(x.size(0), 1, 416, 416)

        return fused
