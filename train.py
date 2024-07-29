import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

from src.trainer.default_trainer import DefaultTrainer
from src.utils.utils import find_file

base_data_path = "data"
base_configs_path = "configs"

if __name__ == "__main__":
    dataset = 'combined'
    dataset_path = "{}/{}".format(base_data_path, dataset)

    config_path = "{}/{}".format(base_configs_path, "unetplusplus_pretrained/version_01.yaml")

    # # if loading pretrained models
    # checkpoints_path = "checkpoints/2024-07-23-23-05-31_SegFormer_KeaggleAugTune_tuned/checkpoints"  # dir where the checkpoint .pt is stored
    # checkpoint_number = 90
    # checkpoints_path = "{}/{}".format(checkpoints_path, checkpoint_number)
    # p1_config_path = find_file(checkpoints_path, ".yaml")
    # p1_pretrained_path = find_file(checkpoints_path, ".pt")

    # checkpoints_path = "checkpoints/2024-07-27-17-27-57_UnetPP_Tune_tuned/checkpoints"  # dir where the checkpoint .pt is stored
    # checkpoint_number = 32
    # checkpoints_path = "{}/{}".format(checkpoints_path, checkpoint_number)
    # p2_config_path = find_file(checkpoints_path, ".yaml")
    # p2_pretrained_path = find_file(checkpoints_path, ".pt")

    ct = DefaultTrainer.from_config(
        config_path,
        config={
            "name": "UnetPP_Pretrain",

            # "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "use_mps": torch.backends.mps.is_available(),
            "wandb": False,
            'train_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "train"), 'type': 'train', 'augment': False},
            'val_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "val"), 'type': 'val'},
            # 'model_config': {'part1_checkpoint_path': p1_pretrained_path, 'part2_checkpoint_path': p2_pretrained_path}
        }
    )
    ct.train()
    # ct.test()
