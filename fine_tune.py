import os

from src.trainer.fine_tune_trainer import TuneTrainer

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

from src.trainer.default_trainer import DefaultTrainer
from src.utils.utils import find_file

base_data_path = "data"
base_configs_path = "configs"
checkpoints_path = "checkpoints/2024-07-16-23-45-13_KeaggleTrainer/checkpoints"  # dir where the checkpoint .pt is stored

if __name__ == "__main__":

    dataset = 'kaegglePure'
    dataset_path = "{}/{}".format(base_data_path, dataset)

    checkpoint_number = 1
    checkpoints_path = "{}/{}".format(checkpoints_path, checkpoint_number)
    config_path = find_file(checkpoints_path, ".yaml")
    pretrained_path = find_file(checkpoints_path, ".pt")

    ct = TuneTrainer.from_config(
        config_path,
        config={
            "name": "SegFormer_KeaggleAugTune",
            "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "use_mps": torch.backends.mps.is_available(),
            "wandb": True,
            'train_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "train"), 'type': 'train', 'augment': True,
                                     "_target_": 'KaeggleDataset'},
            'val_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "val"), 'type': 'val',
                                   "_target_": 'KaeggleDataset'},
            'test_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "test"), 'type': 'test',
                                    "_target_": 'KaeggleDataset'},
            'optimizer_config': {'lr': 0.0004},
            'scheduler_config': {
                'options': {'gamma': 0.9}
            },
            'epochs': 100
        }
    )
    ct.train()

