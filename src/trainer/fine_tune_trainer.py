from typing import Any, Dict, List, Optional

import attr
import torch
from torch import nn
from tqdm import tqdm

# internal
import wandb

from src.base.base_torch_trainer import BaseTorchTrainer
import torch.nn.functional as F


@attr.s(init=False, repr=True)
class TuneTrainer(BaseTorchTrainer):
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
        self.name = self.name + "_tuned"
        self.criterion = nn.BCEWithLogitsLoss()

    def setup_wandb(self):
        if self.wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="satellite-segmentation",
                name=self.name,
                # track hyperparameters and run metadata
                config=self.config
            )

    def visualize(self, out):
        pass

    def load_model(
            self, checkpoint_path: str, load_optimizer_and_scheduler: bool = True
    ):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)
        if "optimizer" in checkpoint:
            if load_optimizer_and_scheduler:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_loss(self, x, targets):
        # Ensure the target has the same shape as output
        # Note: If target is of shape [batchsize, x, y], it needs to be unsqueezed to match [batchsize, 1, x, y]
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if targets.shape[-1] != x.shape[-1]:
            x = F.interpolate(x, size=(416, 416), mode='bilinear', align_corners=False)
        # Compute the loss
        loss = self.criterion(x, targets)

        return loss

    def infer(self, inputs):
        """  
        ## Anuj use in testing // done
        inputs: input satellite images in dataset
        binary_predictions: the predictions binarized to a tensor with 0,1 values. 
        """
        with torch.no_grad():
            predictions = self.model(inputs)
            if predictions.shape[-1] == 104:
                predictions = F.interpolate(predictions, size=(416, 416), mode='bilinear', align_corners=False)

            # print('predictions type', type(predictions)) ## <class 'torch.Tensor'>
            # print('predictions shape', predictions.shape) ## torch.Size([batch, 1, 400, 400])
            binary_predictions = (predictions > 0).float()

            return binary_predictions

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
            "loss": loss.detach().cpu(),
        }
        return out

    def val_func(self, inputs, targets):
        output = self.model(inputs)
        loss = self.get_loss(output, targets)
        out = {
            "out": output,
            "metrics": {"val_loss": loss.item()},
            "loss": loss.detach().cpu(),
        }
        return out

    def train_iter(self, epoch=0):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        current_lr = self.scheduler.get_last_lr()
        with tqdm(self.train_dataloader) as pbar:
            for (_, inputs, labels) in pbar:
                # inputs.to(self.device)
                # labels.to(self.device)

                pbar.set_description(f"Epoch {epoch + 1}/{self.epochs}")

                if self.half_precision:
                    with torch.cuda.amp.autocast():
                        out_dict = self.train_func(inputs, labels)
                else:
                    out_dict = self.train_func(inputs, labels)
                loss, metrics = out_dict["loss"], out_dict["metrics"]

                self.log_step(metrics)

                # if self.visualize_output:
                #     output["prev_batch"].append(inputs.detach().cpu().numpy())
                #     output["post_batch"].append(out.detach().cpu().numpy())
                output["total_metrics"].append(metrics)
                output["total_loss"].append(loss)
                with torch.no_grad():
                    output["loss"] = torch.mean(torch.stack(output["total_loss"]))
                    pbar.set_postfix(loss=loss.item(), total_loss=output["loss"].item(), lr=current_lr)

            for metric in output["total_metrics"][0]:
                output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / len(
                    self.train_dataloader)
            output["loss"] = torch.mean(torch.stack(output["total_loss"]))
            output["metrics"]["lr"] = current_lr[0]
            self.scheduler.step(output["loss"])
            return output

    def val_iter(self, batch_size=32, epoch=0):
        output = {"prev_batch": [], "post_batch": [], "total_metrics": [], "total_loss": [], "metrics": {}}
        with torch.no_grad():
            if len(self.val_dataloader) > 0:
                with tqdm(self.val_dataloader) as pbar:
                    for idx, inputs, labels in pbar:
                        # print()
                        # print(idx)
                        pbar.set_description(f"Epoch {epoch + 1}/{self.epochs} Validation")

                        if self.half_precision:
                            with torch.cuda.amp.autocast():
                                out_dict = self.val_func(inputs, labels)
                        else:
                            out_dict = self.val_func(inputs, labels)
                        loss, metrics = out_dict["loss"], out_dict["metrics"]

                        # if self.visualize_output:
                        #     output["prev_batch"].append(inputs.detach().cpu().numpy())
                        #     output["post_batch"].append(out.detach().cpu().numpy())
                        output["total_metrics"].append(metrics)
                        output["total_loss"].append(loss)
                        output["loss"] = torch.mean(torch.stack(output["total_loss"]))
                        pbar.set_postfix(val_loss=loss.item(), total_val_loss=output["loss"].item())

                    for metric in output["total_metrics"][0]:
                        output["metrics"][metric] = sum([x[metric] for x in output["total_metrics"]]) / len(
                            self.val_dataloader)
                    output["loss"] = torch.mean(torch.stack(output["total_loss"]))
            return output


    def test_iter(self, batch_size=8, epoch=0):
        """
        Main loop for testing. Similar to val_iter but for inference.
        """
        output = {"mask_tensors": [], "file_names": []}

        with torch.no_grad():
            with tqdm(self.test_dataloader) as pbar:
                pbar.set_description(f"Test")
                for step, (file_names, inputs) in enumerate(pbar):
                    if self.half_precision:
                        with torch.cuda.amp.autocast():
                            binary_predictions = self.infer(inputs)
                    else:
                        binary_predictions = self.infer(inputs)

                    # Iterate over the batch and append each prediction and file name individually
                    for prediction, file_name in zip(binary_predictions, file_names):
                        output["mask_tensors"].append(prediction.squeeze())  # remove batch dimension
                        output["file_names"].append(file_name)

        return output
