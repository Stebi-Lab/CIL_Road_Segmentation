import typing
from math import floor
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# internal
from src.base.base_torch_model import BaseTorchModel


class CombiModelConv(BaseTorchModel):
    def __init__(
            self,
            config: Dict[str, Any] = None,
            device: torch.device = None
    ):
        super(CombiModelConv, self).__init__()
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

        self.conv1x1 = nn.Conv2d(2, 1, kernel_size=1)

    def unfreeze_weights(self):
        for param in self.p1.parameters():
            param.requires_grad = True
        for param in self.p2.parameters():
            param.requires_grad = True

    def forward(self, x):
        out1 = self.p1(x)
        if self.print_sizes: print("Out1: ", out1.shape)
        out1 = F.interpolate(out1, size=(400 + self.padding[2], 400 + self.padding[3]), mode='bilinear', align_corners=False)
        if self.print_sizes: print("Out1_interpolated: ", out1.shape)

        out2 = self.p2(x)
        if self.print_sizes: print("Out2: ", out2.shape)

        combined = torch.cat((out1, out2), dim=1)
        if self.print_sizes: print("combined: ", combined.shape)

        fused = self.conv1x1(combined)
        if self.print_sizes: print("fused: ", fused.shape)
        if self.print_sizes: exit()

        return fused
