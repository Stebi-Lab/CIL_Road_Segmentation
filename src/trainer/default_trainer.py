from typing import Any, Dict, List, Optional

import attr
import torch
from torch import nn
from tqdm import tqdm

# internal
import wandb

from src.base.base_torch_trainer import BaseTorchTrainer


@attr.s(init=False, repr=True)
class DefaultTrainer(BaseTorchTrainer):
    use_dataset: bool
    use_model: bool
    half_precision: bool
    torch_seed: Optional[int]
    clip_gradients: bool

    _config_name_: str = "test"

    num_categories: Optional[int] = 0

    def __init__(self, config: Dict[str, Any]):
        self.use_dataset = config["use_dataset"] if "use_dataset" in config.keys() and config[
            "use_dataset"] is not None else True
        self.use_model = config["use_model"] if "use_model" in config.keys() and config[
            "use_model"] is not None else True
        self.half_precision = config["half_precision"] if "half_precision" in config.keys() and config[
            "half_precision"] is not None else False
        self.torch_seed = config["torch_seed"] if "torch_seed" in config.keys() else 0
        self.clip_gradients = config["clip_gradients"] if "clip_gradients" in config.keys() and config[
            "clip_gradients"] is not None else False
        super().__init__(config)

        self.name = "KeaggleTrainer"
        self.criterion = nn.BCEWithLogitsLoss()

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

    def setup_scheduler(self):
        if "_target_" not in self.scheduler_config:
            raise ValueError("No _target_ defined in scheduler_config")
        modules = self.scheduler_config["_target_"].split(".")
        class_name = modules[-1]
        class_ = getattr(torch.optim.lr_scheduler, class_name)
        instance = class_(self.optimizer, factor=self.scheduler_config["factor"],
                          patience=self.scheduler_config["patience"],
                          threshold=self.scheduler_config["threshold"], verbose=True, min_lr=self.scheduler_config["min_lr"])
        self.scheduler = instance

    def visualize(self, out):
        pass

    def get_loss(self, x, targets):
        # Ensure the target has the same shape as output
        # Note: If target is of shape [batchsize, x, y], it needs to be unsqueezed to match [batchsize, 1, x, y]
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        # Compute the loss
        loss = self.criterion(x, targets)

        return loss

    def infer(self, inputs):
        with torch.no_grad():
            return self.model(inputs)

    def train_func(self, inputs, targets):
        self.optimizer.zero_grad()
        output = self.model(inputs)

        loss = self.get_loss(output, targets)

        loss.backward()
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

        self.optimizer.step()
        out = {
            "metrics": {"loss": loss.item()},
            "loss": loss.detach(),
        }
        return out

    def val_func(self, inputs, targets):
        output = self.model(inputs)
        loss = self.get_loss(output, targets)
        out = {
            "out": output,
            "metrics": {"loss": loss.item()},
            "loss": loss,
        }
        return out

    def train_iter(self, batch_size=32, epoch=0):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        with tqdm(self.train_dataloader) as pbar:
            for (inputs, labels) in pbar:
                # inputs.to(self.device)
                # labels.to(self.device)

                pbar.set_description(f"Epoch {epoch+1}/{self.epochs}")

                if self.half_precision:
                    with torch.cuda.amp.autocast():
                        out_dict = self.train_func(inputs, labels)
                else:
                    out_dict = self.train_func(inputs, labels)
                loss, metrics = out_dict["loss"], out_dict["metrics"]
                self.scheduler.step(loss)

                # if self.visualize_output:
                #     output["prev_batch"].append(inputs.detach().cpu().numpy())
                #     output["post_batch"].append(out.detach().cpu().numpy())
                output["total_metrics"].append(metrics)
                output["total_loss"].append(loss)
                with torch.no_grad():
                    output["loss"] = torch.mean(torch.stack(output["total_loss"]))
                    pbar.set_postfix(loss=loss.item(), total_loss=output["loss"].item(), lr=self.scheduler.get_last_lr())

            for metric in output["total_metrics"][0]:
                output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / len(self.train_dataset)
            output["loss"] = torch.mean(torch.stack(output["total_loss"]))
            return output

    def val_iter(self, batch_size=32, epoch=0):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        with torch.no_grad():
            with tqdm(self.val_dataloader) as pbar:
                for inputs, labels in pbar:
                    # inputs.to(self.device)
                    # labels.to(self.device)

                    pbar.set_description(f"Epoch {epoch} Validation")

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
                    output["loss"] = torch.mean(torch.stack(output["total_loss"]))
                    pbar.set_postfix(val_loss=loss.item(), total_val_loss=output["loss"].item())

                for metric in output["total_metrics"][0]:
                    output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / len(self.val_dataset)
                output["loss"] = torch.mean(torch.stack(output["total_loss"]))
                return output

    def test_iter(self, batch_size=32, iteration=0):
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
