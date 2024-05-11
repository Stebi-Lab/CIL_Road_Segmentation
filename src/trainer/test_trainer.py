import random
from enum import Enum
from typing import Any, Dict, List, Optional

import typing
import attr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from IPython.display import clear_output
from matplotlib.colors import hex2color
from tqdm import tqdm
from loguru import logger

# internal
import wandb
from omegaconf import OmegaConf

from src.base.base_torch_trainer import BaseTorchTrainer


@attr.s(init=False, repr=True)
class TestTrainer(BaseTorchTrainer):
    use_dataset: bool
    use_model: bool
    half_precision: bool
    torch_seed: Optional[int]
    clip_gradients: bool

    _config_name_: str = "test"

    num_categories: Optional[int] = 0

    def __init__(self, config: Dict[str, Any]):
        self.use_dataset = config["use_dataset"] if "use_dataset" in config.keys() and config["use_dataset"] is not None else True
        self.use_model = config["use_model"] if "use_model" in config.keys() and config["use_model"] is not None else True
        self.half_precision = config["half_precision"] if "half_precision" in config.keys() and config["half_precision"] is not None else False
        self.torch_seed = config["torch_seed"] if "torch_seed" in config.keys() else 0
        self.clip_gradients = config["clip_gradients"] if "clip_gradients" in config.keys() and config["clip_gradients"] is not None else False
        super().__init__(config)

    def setup_wandb(self):
        if self.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="NCA",
                name=self.name,
                # track hyperparameters and run metadata
                config={
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "num_categories": self.num_categories,
                    "half_precision": self.half_precision
                })

    def visualize(self, out):
        # for tree in range(1):
        #     prev_batch = out["prev_batch"][tree]
        #     post_batch = out["post_batch"][tree]
        #     prev_batch = rearrange(prev_batch, "b d h w c -> b w d h c")
        #     post_batch = rearrange(post_batch, "b d h w c -> b w d h c")
        #     prev_batch = replace_colors(
        #         np.argmax(prev_batch[:, :, :, :, : self.num_categories], -1),
        #         self.dataset.target_color_dict,
        #     )
        #     post_batch = replace_colors(
        #         np.argmax(post_batch[:, :, :, :, : self.num_categories], -1),
        #         self.dataset.target_color_dict,
        #     )
        #     clear_output()
        #     vis0 = prev_batch[:5]
        #     vis1 = post_batch[:5]
        #     num_cols = len(vis0)
        #     vis0[vis0 == "_empty"] = None
        #     vis1[vis1 == "_empty"] = None
        #     print(f'Before --- After --- Tree {tree}')
        #     fig = plt.figure(figsize=(15, 10))
        #     for i in range(1, num_cols + 1):
        #         ax0 = fig.add_subplot(1, num_cols, i, projection="3d")
        #         ax0.voxels(vis0[i - 1], facecolors=vis0[i - 1], edgecolor="k")
        #         ax0.set_title("Index {}".format(i))
        #     for i in range(1, num_cols + 1):
        #         ax1 = fig.add_subplot(2, num_cols, i + num_cols, projection="3d")
        #         ax1.voxels(vis1[i - 1], facecolors=vis1[i - 1], edgecolor="k")
        #         ax1.set_title("Index {}".format(i))
        #     plt.subplots_adjust(bottom=0.005)
        #     plt.show()
        pass

    def get_loss(self, x, targets):
        weight_class = torch.ones(self.num_categories)
        weight_class[0] = 0.01
        class_loss = F.cross_entropy(
            x[:, : self.num_categories, :, :, :], targets, weight=weight_class.to(self.device)
        )
        weight = torch.zeros(self.num_categories)
        weight[0] = 1.0
        alive_loss = F.cross_entropy(
            x[:, : self.num_categories, :, :, :],
            targets,
            weight=weight.to(self.device),
        )

        loss = (0.5 * class_loss + 0.5 * alive_loss) / 2.0
        return loss, class_loss

    def infer(self, inputs):
            with torch.no_grad():
                return self.model(inputs)

    def train_func(self, inputs, targets):
        self.optimizer.zero_grad()
        output = self.model(inputs)

        loss, class_loss = self.get_loss(output, targets)

        loss.backward()
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        self.scheduler.step()
        # x = rearrange(x, "b c d h w -> b d h w c")
        out = {
            "out": output,
            "metrics": {"loss": loss.item(), "class_loss": class_loss.item()},
            "loss": loss,
        }
        return out

    def val_func(self, inputs, targets):
        output = self.model(inputs)
        loss, class_loss = self.get_loss(output, targets)
        out = {
            "out": output,
            "metrics": {"loss": loss.item(), "class_loss": class_loss.item()},
            "loss": loss,
        }
        return out

    def train_iter(self, bar, batch_size=32, iteration=0):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        for step, (inputs, labels) in enumerate(self.train_dataloader):

            if self.half_precision:
                with torch.cuda.amp.autocast():
                    out_dict = self.train_func(inputs, labels)
            else:
                out_dict = self.train_func(inputs, labels)
            _, loss, metrics = out_dict["out"], out_dict["loss"], out_dict["metrics"]

            # if self.visualize_output:
            #     output["prev_batch"].append(inputs.detach().cpu().numpy())
            #     output["post_batch"].append(out.detach().cpu().numpy())
            output["total_metrics"].append(metrics)
            output["total_loss"].append(loss)
            # TODO maybe log individual steps with running loss

        for metric in output["total_metrics"][0]:
            output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / len(self.train_dataset)
        output["loss"] = torch.mean(torch.stack(output["total_loss"]))
        return output

    def val_iter(self, bar, batch_size=32, iteration=0):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(self.val_dataloader):

                if self.half_precision:
                    with torch.cuda.amp.autocast():
                        out_dict = self.val_func(inputs, labels)
                else:
                    out_dict = self.val_func(inputs, labels)
                _, loss, metrics = out_dict["out"], out_dict["loss"], out_dict["metrics"]

                # if self.visualize_output:
                #     output["prev_batch"].append(inputs.detach().cpu().numpy())
                #     output["post_batch"].append(out.detach().cpu().numpy())
                output["total_metrics"].append(metrics)
                output["total_loss"].append(loss)
                # TODO maybe log individual steps with running loss

            for metric in output["total_metrics"][0]:
                output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / len(self.val_dataset)
            output["loss"] = torch.mean(torch.stack(output["total_loss"]))
            return output

    def test_iter(self, bar, batch_size=32, iteration=0):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(self.test_dataloader):

                if self.half_precision:
                    with torch.cuda.amp.autocast():
                        out_dict = self.val_func(inputs, labels)
                else:
                    out_dict = self.val_func(inputs, labels)
                _, loss, metrics = out_dict["out"], out_dict["loss"], out_dict["metrics"]

                # if self.visualize_output:
                #     output["prev_batch"].append(inputs.detach().cpu().numpy())
                #     output["post_batch"].append(out.detach().cpu().numpy())
                output["total_metrics"].append(metrics)
                output["total_loss"].append(loss)
                # TODO maybe log individual steps with running loss

            for metric in output["total_metrics"][0]:
                output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / len(self.val_dataset)
            output["loss"] = torch.mean(torch.stack(output["total_loss"]))
            return output