from __future__ import annotations

import abc
import os
import typing
import wandb
from datetime import datetime
from typing import Any, Dict, Optional

import attr
import numpy as np
import torch
import tqdm
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from src.dataset import datasetMappingDict
from src.model import modelMappingDict
# internal
from src.utils.utils import makedirs, load_config, merge


def fullname(cls):
    module = cls.__module__
    return "{}.{}".format(module, cls.__name__)


@attr.s(init=False, repr=True)
class BaseTorchTrainer(metaclass=abc.ABCMeta):
    _config_group_ = "trainer"

    config: Dict[str, Any]

    name: Optional[str]
    pretrained_path: Optional[str]
    visualize_output: bool
    use_cuda: bool
    use_mps: bool
    device_id: int
    wandb: bool
    early_stoppage: bool
    loss_threshold: float
    batch_size: int
    epochs: int
    checkpoint_interval: int
    verbose: bool

    # config
    model_config: Dict[str, Any]
    dataset_config: Dict[str, Any]
    optimizer_config: Dict[str, Any]
    scheduler_config: Dict[str, Any]
    logging_config: Dict[str, Any]
    dataloader_config: Dict[str, Any]
    # copy config for logging purposes

    # variables that will be populateds
    current_iteration: int
    checkpoint_path: str
    num_samples: Optional[int]

    def __init__(self, config: Dict[str, Any]):
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.run_name = None
        self.base_checkpoint_path = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.checkpoint_setup = False
        self.config = config

        self.name = self.config["name"] if "name" in self.config.keys() else "default"
        self.pretrained_path = self.config["pretrained_path"] if "pretrained_path" in self.config.keys() else None
        self.visualize_output = self.config["visualize_output"] if "visualize_output" in self.config.keys() else False
        self.use_cuda = self.config["use_cuda"] if "use_cuda" in self.config.keys() and self.config[
            "use_cuda"] is not None else False
        self.use_mps = self.config["use_mps"] if "use_mps" in self.config.keys() and self.config[
            "use_mps"] is not None else False
        self.device_id = self.config["device_id"] if "device_id" in self.config.keys() and self.config[
            "device_id"] is not None else 0
        self.early_stoppage = self.config["early_stoppage"] if "early_stoppage" in self.config.keys() and self.config[
            "early_stoppage"] is not None else False
        self.loss_threshold = self.config["loss_threshold"] if "loss_threshold" in self.config.keys() and self.config[
            "loss_threshold"] is not None else 0.002
        self.batch_size = self.config["batch_size"] if "batch_size" in self.config.keys() and self.config[
            "batch_size"] is not None else 32
        self.epochs = self.config["epochs"] if "epochs" in self.config.keys() and self.config[
            "epochs"] is not None else 51
        self.checkpoint_interval = self.config["checkpoint_interval"] if "checkpoint_interval" in self.config.keys() and \
                                                                         self.config[
                                                                             "checkpoint_interval"] is not None else 10
        self.wandb = self.config["wandb"] if "wandb" in self.config.keys() and self.config[
            "wandb"] is not None else False
        self.verbose = self.config["verbose"] if "verbose" in self.config.keys() and self.config[
            "verbose"] is not None else False

        self.model_config = self.config["model_config"] if "model_config" in self.config.keys() and self.config[
            "model_config"] is not None else {}
        self.train_dataset_config = self.config[
            "train_dataset_config"] if "train_dataset_config" in self.config.keys() and self.config[
            "train_dataset_config"] is not None else {}
        self.val_dataset_config = self.config["val_dataset_config"] if "val_dataset_config" in self.config.keys() and \
                                                                       self.config[
                                                                           "val_dataset_config"] is not None else {}
        self.test_dataset_config = self.config["test_dataset_config"] if "test_dataset_config" in self.config.keys() and \
                                                                         self.config[
                                                                             "test_dataset_config"] is not None else {}
        self.optimizer_config = self.config["optimizer_config"] if "optimizer_config" in self.config.keys() and \
                                                                   self.config["optimizer_config"] is not None else {}
        self.scheduler_config = self.config["scheduler_config"] if "scheduler_config" in self.config.keys() and \
                                                                   self.config["scheduler_config"] is not None else {}
        self.logging_config = self.config["logging_config"] if "logging_config" in self.config.keys() and self.config[
            "logging_config"] is not None else {}
        self.dataloader_config = self.config["dataloader_config"] if "dataloader_config" in self.config.keys() and \
                                                                     self.config[
                                                                         "dataloader_config"] is not None else {}

        self.setup_trainer()

    def setup_wandb(self):
        pass

    def setup_trainer(self):
        self.current_iteration = 0
        if self.name is None:
            raise ValueError("Name must be given in config")
        self.config = OmegaConf.to_container(self.config)
        self.config["name"] = self.name

        self.device = torch.device(
            "cuda:{}".format(self.device_id) if self.use_cuda else "mps" if self.use_mps else "cpu"
        )


        self.setup()
        self._setup_dataset()
        self.setup_dataloader()
        self.setup_test_dataloader()
        self._setup_model()
        self._setup_optimizer()
        self.load(self.pretrained_path)
        self.setup_device()
        self.setup_wandb()
        self.post_setup()

    @classmethod
    def from_config(
            cls, config_path: Optional[str] = None, config: Dict[str, Any] = {}
    ) -> BaseTorchTrainer:
        _config = load_config(config_path)
        _config = merge(OmegaConf.create({"trainer": config}), _config)
        print(_config)
        _config["trainer"]["_target_"] = fullname(cls)
        # _config["trainer"]["config"] = _config
        return cls(_config.trainer)

    def setup(self):
        pass

    def setup_logging_and_checkpoints(self):
        if not self.checkpoint_setup:
            self.checkpoint_setup = True
            self.checkpoint_path = makedirs(self.logging_config["checkpoint_path"])
            self.run_name = "{}_{}/".format(
                datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), self.name
            )
            self.base_checkpoint_path = self.make_checkpoint(makedirs(self.checkpoint_path))
            self.checkpoint_path = makedirs(
                os.path.join(self.base_checkpoint_path, "checkpoints")
            )

    # setup helper functions
    def pre_dataset_setup(self):
        pass

    def post_dataset_setup(self):
        pass

    def setup_dataset(self):
        if "_target_" not in self.train_dataset_config:
            raise ValueError("No _target_ defined in dataset_config")
        self.train_dataset = datasetMappingDict[self.train_dataset_config["_target_"]](self.train_dataset_config)
        if "_target_" not in self.val_dataset_config:
            raise ValueError("No _target_ defined in dataset_config")
        self.val_dataset = datasetMappingDict[self.val_dataset_config["_target_"]](self.val_dataset_config)
        if "_target_" not in self.test_dataset_config:
            raise ValueError("No _target_ defined in dataset_config")
        self.test_dataset = datasetMappingDict[self.test_dataset_config["_target_"]](self.test_dataset_config)

    def _setup_dataset(self):
        self.pre_dataset_setup()
        self.setup_dataset()
        self.post_dataset_setup()

    def setup_dataloader(self):
        if "_target_" not in self.dataloader_config:
            raise ValueError("No _target_ defined in dataloader_config")
        modules = self.dataloader_config["_target_"].split(".")
        class_name = modules[-1]
        class_ = getattr(torch.utils.data, class_name)
        instance = class_(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.dataloader_config["num_workers"],
                          shuffle=self.dataloader_config["shuffle"])
        self.train_dataloader = instance
        val_instance = class_(self.val_dataset, batch_size=self.batch_size,
                              num_workers=self.dataloader_config["num_workers"],
                              shuffle=False)
        self.val_dataloader = val_instance

    def setup_test_dataloader(self):
        if "_target_" not in self.dataloader_config:
            raise ValueError("No _target_ defined in dataloader_config")
        modules = self.dataloader_config["_target_"].split(".")
        class_name = modules[-1]
        class_ = getattr(torch.utils.data, class_name)
        test_instance = class_(self.test_dataset, batch_size=self.batch_size,
                               num_workers=self.dataloader_config["num_workers"],
                               shuffle=False)
        self.test_dataloader = test_instance

    def pre_model_setup(self):
        pass

    def post_model_setup(self):
        pass

    def setup_model(self):
        if "_target_" not in self.model_config:
            raise ValueError("No _target_ defined in model_config")
        self.model = modelMappingDict[self.model_config["_target_"]](self.model_config)
        # self.model = instantiate(self.model_config)

    def _setup_model(self):
        self.pre_model_setup()
        self.setup_model()
        self.post_model_setup()

    def pre_optimizer_setup(self):
        pass

    def post_optimizer_setup(self):
        pass

    def setup_optimizer(self):
        if "_target_" not in self.optimizer_config:
            raise ValueError("No _target_ defined in optimizer_config")
        modules = self.optimizer_config["_target_"].split(".")
        class_name = modules[-1]
        class_ = getattr(torch.optim, class_name)
        instance = class_(self.model.parameters(), lr=self.optimizer_config["lr"],
                          weight_decay=self.optimizer_config["weight_decay"], betas=self.optimizer_config["betas"])
        self.optimizer = instance

        # self.optimizer = instantiate(
        #     self.optimizer_config, params=self.model.parameters()
        # )
        # self.scheduler = instantiate(self.scheduler_config, optimizer=self.optimizer)

    def setup_scheduler(self):
        if "_target_" not in self.scheduler_config:
            raise ValueError("No _target_ defined in scheduler_config")
        modules = self.scheduler_config["_target_"].split(".")
        class_name = modules[-1]
        class_ = getattr(torch.optim.lr_scheduler, class_name)
        instance = class_(self.optimizer, **self.scheduler_config["options"])
        self.scheduler = instance

    def _setup_optimizer(self):
        self.pre_optimizer_setup()
        self.setup_optimizer()
        self.setup_scheduler()
        self.post_optimizer_setup()

    def setup_device(self):
        if self.model is not None:
            self.model.device = self.device
            self.model.to(self.device)
        if self.train_dataset is not None:
            self.train_dataset.device = self.device
            self.train_dataset.to_device(self.device)
        if self.val_dataset is not None:
            self.val_dataset.device = self.device
            self.val_dataset.to_device(self.device)
        if self.test_dataset is not None:
            self.test_dataset.device = self.device
            self.test_dataset.to_device(self.device)

    def post_setup(self):
        pass

    def to_device(self, device):
        self.device = device
        self.setup_device()

    def make_checkpoint(self, path: str):
        path = os.path.join(path, self.run_name)
        return path

    def save(
            self,
            base_path: Optional[str] = None,
            step: Optional[int] = None,
            path_name: Optional[str] = None,
    ) -> str:
        if not self.checkpoint_setup:
            self.setup_logging_and_checkpoints()

        checkpoint_path = self.checkpoint_path
        if base_path is not None:
            checkpoint_path = base_path
        if path_name is None:
            if step is None:
                step = self.current_iteration
            path = makedirs("{}/{}".format(checkpoint_path, step))
            torch_path = "{}/{}_iteration_{}.pt".format(path, self.name, step)
        else:
            path = makedirs("{}/{}".format(checkpoint_path, path_name))
            torch_path = "{}/{}.pt".format(path, self.name)
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(checkpoint, torch_path)
        self.save_config(path)  # TODO save metrics
        return path

    def save_config(self, path) -> str:
        _config = OmegaConf.create({"trainer": self.config})
        yaml_str = OmegaConf.to_yaml(_config)
        config_path = os.path.join(path, "{}.yaml".format(self.name))
        with open(config_path, "w") as f:
            f.write(yaml_str)
        return config_path

    def load(self, pretrained_path):
        if pretrained_path is not None:
            self.pretrained_path = os.path.abspath(pretrained_path)
            self.load_model(self.pretrained_path)

    def load_model(
            self, checkpoint_path: str, load_optimizer_and_scheduler: bool = True
    ):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        if "optimizer" in checkpoint and "scheduler" in checkpoint:
            if load_optimizer_and_scheduler:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def log_epoch(self, train_metrics, val_metrics, epoch):
        # for metric in train_metrics:
        #     self.tensorboard_logger.log_scalar(
        #         train_metrics[metric], metric, step=epoch
        #     )
        metrics = train_metrics | val_metrics

        if self.wandb:
            for_wandb = {key: val for key, val in metrics.items()}
            for_wandb["epoch"] = epoch
            wandb.log(for_wandb)

    def log_step(self, train_metrics):
        # for metric in train_metrics:
        #     self.tensorboard_logger.log_scalar(
        #         train_metrics[metric], metric, step=epoch
        #     )
        metrics = train_metrics

        if self.wandb:
            for_wandb = {key: val for key, val in metrics.items()}
            wandb.log(for_wandb)

    def log_test(self, test_metrics):
        # for metric in train_metrics:
        #     self.tensorboard_logger.log_scalar(
        #         train_metrics[metric], metric, step=epoch
        #     )

        if self.wandb:
            for_wandb = {key: val for key, val in test_metrics.items()}
            for_wandb["epoch"] = "test"
            wandb.log(for_wandb)

    def visualize(self, *args, **kwargs):
        """Visualize output
        """
        pass

    @abc.abstractmethod
    def train_iter(
            self, iteration: Optional[int] = None
    ) -> Dict[Any, Any]:
        """
            Training iteration, specify learning process here
        """

    @abc.abstractmethod
    def val_iter(
            self, batch_size: int, iteration: Optional[int] = None
    ) -> Dict[Any, Any]:
        """
            Validation iteration, specify validation process here
        """

    @abc.abstractmethod
    def test_iter(
            self, batch_size: int, iteration: Optional[int] = None
    ) -> Dict[Any, Any]:
        """
            Test iteration, specify testing process here
        """

    def pre_train(self):
        pass

    def post_train(self):
        pass

    def pre_train_iter(self):
        self.model.train()

    def post_train_iter(self, train_output: Dict[Any, Any]):
        pass

    def pre_val_iter(self):
        self.model.eval()

    def post_val_iter(self, val_output: Dict[Any, Any]):
        pass

    def pre_test_iter(self):
        self.model.eval()

    def post_test_iter(self, test_output: Dict[Any, Any]):
        pass

    def train(
            self, batch_size=None, epochs=None, checkpoint_interval=None, visualize=None
    ) -> typing.Dict[str, Any]:
        """Main training function, should call train_iter
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if epochs is not None:
            self.epochs = epochs
        if checkpoint_interval is not None:
            self.checkpoint_interval = checkpoint_interval
        self.pre_train()
        self.setup_dataloader()
        for i in range(self.epochs):
            self.pre_train_iter()
            output = self.train_iter(i)
            self.post_train_iter(output)
            metrics = output.get("metrics", {})
            loss = output["loss"]

            self.pre_val_iter()
            val_output = self.val_iter(self.batch_size, i)
            self.post_val_iter(val_output)
            val_metrics = val_output.get("metrics", {})

            self.log_epoch(metrics, val_metrics, i)

            if i % self.checkpoint_interval == 0:
                if self.visualize_output:
                    self.visualize(output, val_output)
                self.save(step=i)

            if self.early_stoppage:
                if loss <= self.loss_threshold:
                    self.save(step=i)
                    break
        self.post_train()
        return metrics, val_metrics

    def test(self, batch_size=None, visualize=None) -> typing.Dict[str, Any]:
        """Main testing function, should call test_iter."""

        if batch_size is not None:
            self.batch_size = batch_size
        self.setup_test_dataloader()
        self.pre_test_iter()
        output = self.test_iter(self.batch_size)

        self.post_test_iter(output)
        mask_tensors = output.get("mask_tensors", [])
        file_names = output.get("file_names", [])

        # Save PNG images to folder
        self.save_to_image(mask_tensors, file_names, save_dir='results')

        return output

    def save_to_image(self, mask_tensors, file_names, save_dir='results'):
        """
        Saves the masks to the save_dir directory as png images.
        """

        # Create save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save each binary mask as a PNG image
        for mask_tensor, file_name in zip(mask_tensors, file_names):
            mask_array = mask_tensor.cpu().numpy().astype(np.uint8) * 255  # Convert to uint8 and scale to [0, 255]
            if mask_array.ndim == 2:
                mask_image = Image.fromarray(mask_array)
            elif mask_array.ndim == 3:
                mask_image = Image.fromarray(mask_array[0])  # Handle (1, H, W) shape
            else:
                raise ValueError(f"Unexpected mask array shape: {mask_array.shape}")
            mask_image.save(os.path.join(save_dir, file_name))  # Save with the correct filename

        return None
