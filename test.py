import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

from src.trainer.default_trainer import DefaultTrainer
from src.utils.utils import find_file

from absl import app, flags
from mask_to_submission import main as mask_to_submission_main

base_data_path = "data"
base_configs_path = "configs"
checkpoints_path = "checkpoints/2024-07-28-15-00-03_Combined_MLP/checkpoints"  # TODO: input correct checkpoint path to test

if __name__ == "__main__":
    dataset = 'kaegglePure'
    dataset_path = "{}/{}".format(base_data_path, dataset)

    # TODO: Choose the best checkpoint
    checkpoint_number = 30
    checkpoints_path = "{}/{}".format(checkpoints_path, checkpoint_number)
    config_path = find_file(checkpoints_path, ".yaml")
    pretrained_path = find_file(checkpoints_path, ".pt")

    ct = DefaultTrainer.from_config(
        config_path,
        config={
            "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "use_mps": torch.backends.mps.is_available(),
            "wandb": False,
            'train_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "train"), 'type': 'train', 'augment': False,
                                     "_target_": 'KaeggleDataset'},
            'val_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "val"), 'type': 'val',
                                   "_target_": 'KaeggleDataset'},
            'test_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "test"), 'type': 'test',
                                    "_target_": 'KaeggleDataset'},
        }
    )

    ct.test()  # predicts masks and stores PNGs in results folder
    app.run(mask_to_submission_main)  # creates submission.csv. Args in mask_to_submission.py
