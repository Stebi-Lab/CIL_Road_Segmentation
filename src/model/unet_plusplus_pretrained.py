import segmentation_models_pytorch as smp
import torch
from torch import nn
from src.base.base_torch_model import BaseTorchModel
from typing import Any, Dict, List, Optional


class UNetPlusPlusModel_Pretrained(BaseTorchModel):
    def __init__(
            self,
            config: Dict[str, Any] = None,
    ):
        super(UNetPlusPlusModel_Pretrained, self).__init__()
        self.config = config
        self.network_shape = self.config["network_shape"] if "network_shape" in self.config.keys() and self.config[
            "network_shape"] is not None else [[64, 3, 3, 2], [128, 3, 2, 2], [256, 3, 1, 2]]

        self.num_channels = 3

        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config[
            "verbose"] is not None else False

        self.print_sizes = self.config["print_sizes"] if "print_sizes" in self.config.keys() and self.config[
            "print_sizes"] is not None else False
        
        if self.print_sizes:
            print()
            print("DEBUG MODE: printing sizes for one iteration")
            print()

        input_size = 400

        nb_filter = [32, 64, 128, 256, 512]

        num_classes = 1

        self.model = smp.UnetPlusPlus(
                 encoder_name='efficientnet-b7',
                 encoder_weights='imagenet',
                 in_channels=3,
                 classes=1)



    def forward(self, input):
        output = self.model.forward(input)
        return output