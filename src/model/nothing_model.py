import typing
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

# internal
from src.base.base_torch_model import BaseTorchModel


def make_sequental(num_channels, channel_dims, embedding_dim=None):
    conv3d = torch.nn.Conv3d(num_channels * 3 + (embedding_dim if embedding_dim else 0), channel_dims[0], kernel_size=1)
    gelu = torch.nn.GELU()
    layer_list = [conv3d, gelu]

    for i in range(1, len(channel_dims)):
        layer_list.append(
            torch.nn.Conv3d(channel_dims[i - 1], channel_dims[i], kernel_size=1)
        )
        layer_list.append(torch.nn.GELU())
    layer_list.append(
        torch.nn.Conv3d(channel_dims[-1], num_channels, kernel_size=1, bias=False)
    )
    return torch.nn.Sequential(*layer_list)


class VoxelPerceptionNet(torch.nn.Module):
    def __init__(
            self, num_channels=1, normal_std=0.02, use_normal_init=True, zero_bias=True
    ):
        super(VoxelPerceptionNet, self).__init__()
        self.num_channels = num_channels
        self.normal_std = normal_std
        self.conv1 = torch.nn.Conv3d(
            self.num_channels,
            self.num_channels * 3,
            3,
            stride=1,
            padding=1,
            groups=self.num_channels,
            bias=False,
        )

        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=normal_std)
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.normal_(m.bias, std=normal_std)

        if use_normal_init:
            with torch.no_grad():
                self.apply(init_weights)

    def forward(self, x):
        return self.conv1(x)


class SmallerVoxelUpdateNet(torch.nn.Module):
    def __init__(
            self,
            num_channels: int = 16,
            channel_dims=[64, 64],
            normal_std=0.02,
            use_normal_init=True,
            zero_bias=True,
            embedding_dim: Optional[int] = None,  # new

    ):
        super(SmallerVoxelUpdateNet, self).__init__()
        self.embedding_dim = embedding_dim

        self.out = make_sequental(num_channels, channel_dims, embedding_dim)

        def init_weights(m):
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.normal_(m.weight, std=normal_std)
                if getattr(m, "bias", None) is not None:
                    if zero_bias:
                        torch.nn.init.zeros_(m.bias)
                    else:
                        torch.nn.init.normal_(m.bias, std=normal_std)

        if use_normal_init:
            with torch.no_grad():
                self.apply(init_weights)

    def forward(self, x, emeddings=None):
        if emeddings != None:
            # concat with embeddings
            x = torch.cat((x, emeddings), 1)  # batch, embedd+channel, cordinates(3d)
        return self.out(x)


class TestModel(BaseTorchModel):
    def __init__(
            self,
            config: Dict[str, Any] = None,
    ):
        super(TestModel, self).__init__()
        self.config = config
        self.update_net_channel_dims = self.config["update_net_channel_dims"] if "update_net_channel_dims" in self.config.keys() and self.config["update_net_channel_dims"] is not None else [32, 32]
        self.num_channels = 1
        self.perception_net = VoxelPerceptionNet(
            self.num_channels,
        )
        self.update_network = SmallerVoxelUpdateNet(
            self.num_channels,
        )
        self.tanh = torch.nn.Tanh()
        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config["verbose"] is not None else False

    def alive(self, x):
        return F.max_pool3d(
            x[:, self.living_channel_dim: self.living_channel_dim + 1, :, :, :],
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def perceive(self, x):
        return self.perception_net(x)

    def update(self, x, embeddings=None):
        pre_life_mask = self.alive(x) > self.alpha_living_threshold

        out = self.perceive(x)
        out = self.update_network(out, embeddings)

        rand_mask = torch.rand_like(x[:, :1, :, :, :]) < self.cell_fire_rate
        out = out * rand_mask.float().to(self.device)
        x = x + out

        post_life_mask = self.alive(x) > self.alpha_living_threshold
        life_mask = (pre_life_mask & post_life_mask).float()
        x = x * life_mask
        if not self.use_bce_loss:
            x[:, :1, :, :, :][life_mask == 0.0] += torch.tensor(1.0).to(self.device)
        return x, life_mask

    def forward(self, x, steps=1, embeddings=None, rearrange_output=True, return_life_mask=False):
        x = rearrange(x, "b d h w c -> b c d h w")
        if embeddings != None:
            embeddings = rearrange(embeddings, "b d h w c -> b c d h w")
        for step in range(steps):
            x, life_mask = self.update(x, embeddings=embeddings)
        if rearrange_output:
            x = rearrange(x, "b c d h w -> b d h w c")
        if return_life_mask:
            return x, life_mask
        return x
