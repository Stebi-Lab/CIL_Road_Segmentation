import torch

from src.trainer.default_trainer import DefaultTrainer

base_data_path = "data"
base_configs_path = "configs"

if __name__ == "__main__":

    dataset = 'kaeggle'
    dataset_path = "{}/{}".format(base_data_path, dataset)

    # if loading pretrained model
    # checkpoints_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2024-01-05-00-40-39_VariousTrees_20tree_final_pink/checkpoints"
    # directory = f"{checkpoints_path}/{2600}"
    # config_path = find_file(directory, ".yaml")
    # pretrained_path = find_file(directory, ".pt")

    config_path = "{}/{}".format(base_configs_path, "unet/version_01.yaml")
    ct = DefaultTrainer.from_config(
        config_path,
        config={
            # "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "wandb": False,
            'train_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "train"), "preload_all": False},
            'val_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "val"), "preload_all": False},
        }
    )
    ct.train()
