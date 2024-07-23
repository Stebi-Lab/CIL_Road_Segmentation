import os
from typing import Any, Dict
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

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
        self.padding = self.config["padding"] if "padding" in self.config.keys() and self.config[
            "padding"] is not None else [0, 0, 0, 0]
        self.type = self.config["type"] if "type" in self.config.keys() and self.config[
            "type"] is not None else 'train'
        self.augment = self.config["augment"] if "augment" in self.config.keys() and self.config[
            "augment"] is not None else False

        if self.verbose: print("Padding with {}".format(tuple(self.padding)))

        self.device = 'cpu'
        self.targets = []
        self.data = []
        self.num_samples = 0

        if self.verbose:
            print('self.augment is:', self.augment)
            print('self.type is:', self.type)

        self.sample_transforms = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Pad(tuple(self.padding))
        ])
        self.label_transforms = transforms.Compose([
            # transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Pad(tuple(self.padding))
        ])

        if self.dataset_path is not None:
            imgs_path = self.dataset_path + "/images"
            labels_path = self.dataset_path + "/labels"
            if not os.path.exists(imgs_path):
                raise Exception("No images folder found")
            samples = sorted([name for name in os.listdir(imgs_path) if '.png' in name])
            num_samples = len(samples)
            if self.verbose: print("Found {} samples at path: {}".format(num_samples, imgs_path))

            self.data = [imgs_path + "/" + file for file in samples]

            if (self.type != 'test') and os.path.exists(labels_path):
                labels = sorted([name for name in os.listdir(labels_path) if '.png' in name])
                if len(labels) != num_samples: raise Exception("Number of labels does not match number of samples")

                self.targets = [labels_path + "/" + file for file in labels]

            self.num_samples = num_samples

        if self.verbose:
            print(f"Loaded dataset {self.entity_name} with {len(self)} samples")

    def to_device(self, device):
        self.device = device

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.type != 'test':
            img, label = self.load_both(idx)
            return self.data[idx].split('/')[-1], img.to(self.device), label.to(self.device)
        else:
            return self.data[idx].split('/')[-1], self.load_single_img(self.data[idx]).to(self.device)

    def load_single_img(self, file_path):
        image = Image.open(file_path).convert("RGB")

        if self.sample_transforms:
            image = self.sample_transforms(image)
        return image

    def load_both(self, idx):
        """
        Loads both Image and Label concurrently and transforms them
        """

        img_file_path = self.data[idx]
        label_file_path = self.targets[idx]

        image = Image.open(img_file_path).convert("RGB")
        label = Image.open(label_file_path).convert("L")

        if self.sample_transforms:
            image = self.sample_transforms(image)
        if self.label_transforms:
            label = self.label_transforms(label)

        # Save for visualization
        # self.save_img_label_png(idx, image, label, dir='before_augment')

        # Apply augmentations
        if self.augment and self.type == 'train':
            severity = 0.4  # Adjust severity as needed
            image, label = self.random_augmentations(image, label, severity)

        # Save for visualization
        # self.save_img_label_png(idx, image, label, dir='after_augment')

        return image, label

    def random_augmentations(self, image, label, severity):

        # Random horizontal flip
        if random.random() > 0.5:
            image = F.hflip(image)
            label = F.hflip(label)

        # Random vertical flip
        if random.random() > 0.5:
            image = F.vflip(image)
            label = F.vflip(label)

        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        image = F.rotate(image, angle)
        label = F.rotate(label, angle)

        # Random zoom and crop
        scale = random.uniform(1 - severity, 1 + severity)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=[scale, scale], ratio=[1.0, 1.0]
        )
        image = F.resized_crop(image, i, j, h, w, size=[image.shape[1], image.shape[2]])
        label = F.resized_crop(label, i, j, h, w, size=[label.shape[1], label.shape[2]])

        return image, label

    def save_img_label_png(self, idx, img, label, dir='augment_examples'):
        """
        Save combined image and label for visual inspection.
        """
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Convert tensors to PIL images
        img = transforms.ToPILImage()(img)
        label = transforms.ToPILImage()(label)

        # Concatenate image and label horizontally
        combined_img = Image.new('RGB', (img.width + label.width, img.height))
        combined_img.paste(img, (0, 0))
        combined_img.paste(label, (img.width, 0))

        # Save the image
        combined_img.save(os.path.join(dir, f'{idx}.png'))
