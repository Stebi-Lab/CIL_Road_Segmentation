import typing
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

# internal
from src.base.base_torch_model import BaseTorchModel


class UnetModel(BaseTorchModel):
    def __init__(
            self,
            config: Dict[str, Any] = None,
    ):
        super(UnetModel, self).__init__()
        self.config = config
        self.update_net_channel_dims = self.config["update_net_channel_dims"] if "update_net_channel_dims" in self.config.keys() and self.config["update_net_channel_dims"] is not None else [32, 32]
        self.num_channels = 1
        self.tanh = torch.nn.Tanh()
        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config["verbose"] is not None else False

        U_Shape = [[64, 5, 2, 2], [128, 3, 3, 2], [256, 3, 3, 2]]  # [[out_channels_1,Kernelsize_1,ConvSteps_1,Poolingsize_1],...]
        conv_down_Layers = []
        conv_up_Layers = []
        pooling_Layers = []
        unpooling_Layers = []
        self.Num_Etages = len(U_Shape)
        for Etage_Number, Etage in enumerate(U_Shape):

            Etage_Layers_down = []
            Etage_Layers_up = []

            if Etage[2] > 0:

                if Etage_Number == 0:
                    Etage_Layers_down.append(nn.Conv2d(3, Etage[0], Etage[1]))
                else:
                    Etage_Layers_down.append(nn.Conv2d(U_Shape[Etage_Number - 1][0], Etage[0], Etage[1]))

                for ACS in range(Etage[2] - 1):
                    Etage_Layers_down.append(nn.Conv2d(Etage[0], Etage[0], Etage[1]))
                    if ACS == 0 and self.Num_Etages - 1 != Etage_Number:
                        Etage_Layers_up.append(nn.ConvTranspose2d(Etage[0] * 2, Etage[0], Etage[1]))
                    else:
                        Etage_Layers_up.append(nn.ConvTranspose2d(Etage[0], Etage[0], Etage[1]))

                if Etage_Number == 0:
                    if Etage[2] == 1 and self.Num_Etages - 1 != Etage_Number:
                        Etage_Layers_up.append(nn.ConvTranspose2d(Etage[0] * 2, 1, Etage[1]))
                    else:
                        Etage_Layers_up.append(nn.ConvTranspose2d(Etage[0], 1, Etage[1]))
                else:
                    if Etage[2] == 1 and self.Num_Etages - 1 != Etage_Number:
                        Etage_Layers_up.append(
                            nn.ConvTranspose2d(Etage[0] * 2, U_Shape[Etage_Number - 1][0], Etage[1]))
                    else:
                        Etage_Layers_up.append(nn.ConvTranspose2d(Etage[0], U_Shape[Etage_Number - 1][0], Etage[1]))
            else:
                print("[ERROR]", flush=True)

            conv_down_Layers.append(nn.ModuleList(Etage_Layers_down))
            conv_up_Layers = [nn.ModuleList(Etage_Layers_up)] + conv_up_Layers

            if self.Num_Etages - 1 != Etage_Number:
                pooling_Layers.append(nn.MaxPool2d(Etage[3], return_indices=True))
                unpooling_Layers = [nn.MaxUnpool2d(Etage[3])] + unpooling_Layers

        self.conv_down_Layers = nn.ModuleList(conv_down_Layers)
        self.conv_up_Layers = nn.ModuleList(conv_up_Layers)
        self.pooling_Layers = nn.ModuleList(pooling_Layers)
        self.unpooling_Layers = nn.ModuleList(unpooling_Layers)
        if self.verbose:
            print(self.conv_down_Layers)
            print(self.conv_up_Layers)
            print(self.pooling_Layers)
            print(self.unpooling_Layers)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):

        skipConnections = []
        indices_Arr = []

        for Etage_Number, Etage in enumerate(self.conv_down_Layers):
            for Etage_Layer in Etage:
                x = Etage_Layer(x)
                x = F.relu(x)
            if self.Num_Etages - 1 != Etage_Number:
                skipConnections = [x] + skipConnections
                x, indices = self.pooling_Layers[Etage_Number](x)
                indices_Arr = [indices] + indices_Arr
            # print(x.shape)

        for Etage_Number, Etage in enumerate(self.conv_up_Layers):

            channels = x.shape[-3]
            if Etage_Number != 0:
                channels += skipConnections[Etage_Number - 1].shape[-3]
                x = torch.cat((skipConnections[Etage_Number - 1], x), -3)

            for Etage_Layer in Etage:
                x = F.relu(x)
                x = Etage_Layer(x)

            if self.Num_Etages - 1 != Etage_Number:
                x = self.unpooling_Layers[Etage_Number](x, indices_Arr[Etage_Number])
            # print(x.shape)

        return self.sig(x)
