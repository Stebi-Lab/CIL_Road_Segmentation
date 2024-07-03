import torch

from src.trainer.default_trainer import DefaultTrainer
from src.utils.utils import find_file

from absl import app, flags
from mask_to_submission import main as mask_to_submission_main


base_data_path = "data"                 ## dir where the train/val/test data is stored
base_configs_path = "configs"           ## dir where the configuration .yaml is stored
checkpoints_path = "checkpoints/100/"   ## dir where the checkpoint .pt is stored

if __name__ == "__main__":

    dataset = 'kaeggle'
    dataset_path = "{}/{}".format(base_data_path, dataset)
    config_path = find_file(checkpoints_path, ".yaml")
    pretrained_path = find_file(checkpoints_path, ".pt")

    ct = DefaultTrainer.from_config(
        config_path,
        config={
            "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "wandb": False,
            'train_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "train"), },
            'val_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "val"), },
            'test_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "test"), 'test': True, "_target_" : 'KaeggleDataset'},
        }
    )

    ct.test()                           ## predicts masks and stores PNGs in results folder
    app.run(mask_to_submission_main)    ## creates submission.csv. Args in mask_to_submission.py



