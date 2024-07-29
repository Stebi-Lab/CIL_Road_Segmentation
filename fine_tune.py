import os
from src.trainer.fine_tune_trainer import TuneTrainer

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch
from src.utils.utils import find_file
from absl import app
from mask_to_submission import main as mask_to_submission_main

base_data_path = "data"
base_configs_path = "configs"
checkpoints_path = "checkpoints/2024-07-27-17-27-57_UnetPP_Tune_tuned/checkpoints"  # TODO: input correct checkpoint path to continue from

if __name__ == "__main__":
    dataset = 'kaegglePure'
    dataset_path = "{}/{}".format(base_data_path, dataset)

    # TODO: Choose best checkpoint number
    checkpoint_number = 32
    checkpoints_path = "{}/{}".format(checkpoints_path, checkpoint_number)
    config_path = find_file(checkpoints_path, ".yaml")
    pretrained_path = find_file(checkpoints_path, ".pt")

    ct = TuneTrainer.from_config(
        config_path,
        config={
            "name": "UnetPP_Final",
            "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "use_mps": torch.backends.mps.is_available(),
            "wandb": False,
            'train_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "train"), 'type': 'train',
                                     'augment': True,
                                     "_target_": 'KaeggleDataset'},
            'val_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "val"), 'type': 'val',
                                   "_target_": 'KaeggleDataset'},
            'test_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "test"), 'type': 'test',
                                    "_target_": 'KaeggleDataset'},
            'optimizer_config': {'lr': 0.001},
            'scheduler_config': {
                '_target_': 'torch.optim.lr_scheduler.StepLR',
                'options': {'gamma': 0.5, 'step_size': 20}  # TODO adjust lr schedule if necessary
            },
            'epochs': 40  # TODO adjust epochs if necessary
        }
    )
    ct.train()
