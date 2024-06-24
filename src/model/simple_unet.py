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
        # [[out_channels_1,Kernelsize_1,ConvSteps_1,Poolingsize_1],...]
        self.network_shape = self.config["network_shape"] if "network_shape" in self.config.keys() and self.config[
            "network_shape"] is not None else [[64, 5, 2, 2], [128, 3, 3, 2], [256, 3, 3, 2]]

        self.num_channels = 1
        self.tanh = torch.nn.Tanh()
        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config[
            "verbose"] is not None else False

        conv_down_Layers = []
        conv_up_Layers = []
        pooling_Layers = []
        unpooling_Layers = []
        self.Num_Etages = len(self.network_shape)
        for Etage_Number, Etage in enumerate(self.network_shape):

            Etage_Layers_down = []
            Etage_Layers_up = []

            if Etage[2] > 0:

                if Etage_Number == 0:
                    Etage_Layers_down.append(nn.Conv2d(3, Etage[0], Etage[1]))
                else:
                    Etage_Layers_down.append(nn.Conv2d(self.network_shape[Etage_Number - 1][0], Etage[0], Etage[1]))

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
                            nn.ConvTranspose2d(Etage[0] * 2, self.network_shape[Etage_Number - 1][0], Etage[1]))
                    else:
                        Etage_Layers_up.append(
                            nn.ConvTranspose2d(Etage[0], self.network_shape[Etage_Number - 1][0], Etage[1]))
            else:
                print("[ERROR] Unet configuration wrong")

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
            print(self.network_shape, " => ")
            print("Down layers: ", self.conv_down_Layers)
            print("Up layers: ", self.conv_up_Layers)
            print("Pooling layers: ", self.pooling_Layers)
            print("Unpooling layers: ", self.unpooling_Layers)
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

        return x
