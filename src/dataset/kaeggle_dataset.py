import os
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from einops import repeat, rearrange
from torchvision import transforms

from src.base.base_torch_dataset import BaseTorchDataset


class KaeggleDataset(BaseTorchDataset):
    def __init__(
            self,
            config: Dict[str, Any],
    ):
        self.config = config
        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config[
            "verbose"] is not None else False
        self.entity_name = self.config["entity_name"] if "entity_name" in self.config.keys() and self.config[
            "entity_name"] is not None else "Kaeggle-Dataset"
        self.dataset_path = self.config["dataset_path"] if "dataset_path" in self.config.keys() else None
        self.preloadAll = self.config["preload_all"] if "preload_all" in self.config.keys() and self.config[
            "preload_all"] is not None else False
        self.device = 'cpu'
        self.targets = []
        self.data = []
        self.num_samples = 0
        self.sample_transforms = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_transforms = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        if self.dataset_path is not None:
            imgs_path = self.dataset_path + "/images"
            labels_path = self.dataset_path + "/labels"
            if not os.path.exists(imgs_path):
                raise Exception("No images folder found")
            samples = [name for name in os.listdir(imgs_path) if '.png' in name]
            num_samples = len(samples)
            if self.verbose: print("Found {} samples at path: {}".format(num_samples, imgs_path))
            if self.preloadAll:
                for image in samples:
                    self.data.append(self.load_single_img(imgs_path + "/" + image))
            else:
                self.data = [imgs_path + "/" + file for file in samples]
            if os.path.exists(labels_path):
                labels = [name for name in os.listdir(labels_path) if '.png' in name]
                if len(labels) != num_samples: raise Exception("Number of labels does not match number of samples")
                if self.preloadAll:
                    for image in labels:
                        self.targets.append(self.load_single_img_label(labels_path + "/" + image))
                else:
                    self.targets = [labels_path + "/" + file for file in labels]

            self.num_samples = num_samples

        if self.verbose:
            print(f"Loaded dataset {self.entity_name} with {len(self)} samples")
            if self.preloadAll:
                print(f"with shape {self.data[0].shape} and labels with shape {self.targets[0].shape}")
        self.half_precision = self.config["half_precision"] if "half_precision" in self.config.keys() and self.config[
            "half_precision"] is not None else False

        # if self.half_precision:
        #     self.data = [s.astype(np.float16) for s in self.data]
        # else:
        #     self.data = [s.astype(np.float32) for s in self.data]

    def to_device(self, device):
        self.device = device
        self.data = [t.to(device) for t in self.data]
        self.targets = [t.to(device) for t in self.targets]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.preloadAll:
            return self.data[idx], self.targets[idx]
        return self.load_single(idx)

    def load_single(self, idx):
        return self.load_single_img(self.data[idx]).to(self.device), self.load_single_img_label(self.targets[idx]).to(self.device)

    def load_single_img(self, file_path):
        image = Image.open(file_path).convert("RGB")

        if self.sample_transforms:
            image = self.sample_transforms(image)
        return image

    def load_single_img_label(self, file_path):
        image = Image.open(file_path).convert("L")

        if self.label_transforms:
            image = self.label_transforms(image)
        return image
