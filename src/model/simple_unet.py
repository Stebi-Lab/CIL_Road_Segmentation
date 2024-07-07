import typing
from math import floor
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
            "network_shape"] is not None else [[64, 3, 3, 2], [128, 3, 2, 2], [256, 3, 1, 2]]

        self.num_channels = 1

        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config[
            "verbose"] is not None else False

        self.print_sizes = self.config["print_sizes"] if "print_sizes" in self.config.keys() and self.config[
            "print_sizes"] is not None else False

        conv_down_layers = []
        conv_up_layers = []
        pooling_layers = []
        unpooling_layers = []
        self.num_floors = len(self.network_shape)

        if self.print_sizes:
            print()
            print("DEBUG MODE: printing sizes for one iteration")
            print()

        input_size = 400
        # need_padding = [False for _ in range(self.num_floors)]
        for floor_number, f_config in enumerate(self.network_shape):

            floor_convs_down = []
            floor_convs_up = []

            first_floor = floor_number == 0
            bottom_floor = floor_number == self.num_floors - 1
            if self.print_sizes: print("Start size: ", input_size)
            if f_config[2] > 1:  # Num convolutions

                # First conv per floor
                if first_floor:
                    floor_convs_down.append(nn.Conv2d(3, f_config[0], f_config[1]))
                    input_size = input_size + 2 * 0 - 1 * (f_config[1] - 1)
                    if self.print_sizes: print("Next size: ", input_size)

                else:
                    floor_convs_down.append(
                        nn.Conv2d(self.network_shape[floor_number - 1][0], f_config[0], f_config[1]))
                    input_size = input_size + 2 * 0 - 1 * (f_config[1] - 1)
                    if self.print_sizes: print("Next size: ", input_size)

                num_additional = f_config[2] - 1
                if bottom_floor or first_floor:
                    num_additional -= 1
                for additional_conf in range(num_additional):
                    floor_convs_down.append(nn.Conv2d(f_config[0], f_config[0], f_config[1]))
                    input_size = input_size + 2 * 0 - 1 * (f_config[1] - 1)
                    if self.print_sizes: print("Next size: ", input_size)

                if not bottom_floor:
                    if floor(input_size / f_config[3]) * f_config[3] != input_size and not self.print_sizes:
                        raise ValueError(
                            "Tensor sizes are not evenly divisible with input size {} and pooling size {}! Pooling size and cumulative convolution size decrease give odd input size! For now not supported! Debug with print_sizes flag".format(input_size, f_config[3]))
                        # need_padding[floor_number] = True
                        # input_size += f_config[3]

                    input_size /= f_config[3]
                    if self.print_sizes: print("After pool size: ", input_size)


                if not bottom_floor:
                    floor_convs_up.append(nn.ConvTranspose2d(f_config[0] * 2, f_config[0], f_config[1]))

                for additional_conf in range(f_config[2] - 2):
                    floor_convs_up.append(nn.ConvTranspose2d(f_config[0], f_config[0], f_config[1]))

                if first_floor:
                    floor_convs_up.append(nn.ConvTranspose2d(f_config[0], 1, 1))
                else:
                    floor_convs_up.append(
                        nn.ConvTranspose2d(f_config[0], self.network_shape[floor_number - 1][0], f_config[1]))

            else:
                raise ValueError(
                    "Floor must have at least 2 convolutions, not {}!".format(f_config[2]))

            conv_down_layers.append(nn.ModuleList(floor_convs_down))
            conv_up_layers = [nn.ModuleList(floor_convs_up)] + conv_up_layers

            if self.num_floors - 1 != floor_number:
                pooling_layers.append(nn.MaxPool2d(f_config[3], return_indices=True))
                unpooling_layers = [nn.MaxUnpool2d(f_config[3])] + unpooling_layers

        self.conv_down_layers = nn.ModuleList(conv_down_layers)
        self.conv_up_layers = nn.ModuleList(conv_up_layers)
        self.pooling_layers = nn.ModuleList(pooling_layers)
        self.unpooling_layers = nn.ModuleList(unpooling_layers)
        self.dropout = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout2d(p=0.2)

        if self.verbose:
            print(self.network_shape, " => ")
            print("Down layers: ", self.conv_down_layers)
            print("Up layers: ", self.conv_up_layers)
            print("Pooling layers: ", self.pooling_layers)
            print("Unpooling layers: ", self.unpooling_layers)

    def forward(self, x):



        skipConnections = []
        indices_Arr = []
        if self.print_sizes: print()
        if self.print_sizes: print("Start: ", x.shape)
        for Etage_Number, Etage in enumerate(self.conv_down_layers):
            for Etage_Layer in Etage:
                x = self.dropout(x)
                x = Etage_Layer(x)
                x = F.relu(x)
                if self.print_sizes: print("Down "+str(Etage_Number)+": ", x.shape)
            if self.num_floors - 1 != Etage_Number:
                skipConnections = [x] + skipConnections
                x, indices = self.pooling_layers[Etage_Number](x)
                if self.print_sizes: print("Down pool "+str(Etage_Number)+": ", x.shape)
                indices_Arr = [indices] + indices_Arr

        if self.print_sizes: print("Bottom: ", x.shape)
        for Etage_Number, Etage in enumerate(self.conv_up_layers):

            channels = x.shape[-3]

            if Etage_Number != 0:
                channels += skipConnections[Etage_Number - 1].shape[-3]
                x = torch.cat((skipConnections[Etage_Number - 1], x), -3)
                if self.print_sizes: print("Concat " + str(Etage_Number) + ": ", x.shape, channels)
            with torch.no_grad():

                for Etage_Layer in Etage:
                    x = F.relu(x)
                    x = Etage_Layer(x)
                    if self.print_sizes: print("Up "+str(Etage_Number)+": ", x.shape)

            if self.num_floors - 1 != Etage_Number:
                x = self.unpooling_layers[Etage_Number](x, indices_Arr[Etage_Number])
                if self.print_sizes: print("Up pool "+str(Etage_Number)+": ", x.shape)

        if self.print_sizes: print("Final: ", x.shape)
        if self.print_sizes: exit()

        return x
