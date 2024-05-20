import torch

from src.trainer.test_trainer import KeaggleTrainer

base_data_path = "C:/Users/cedri/Desktop/Code/ETH/cil-road-segmentation/data"

if __name__ == "__main__":

    dataset = 'kaeggle'
    dataset_path = "{}/{}".format(base_data_path, dataset)

    # if loading pretrained model
    # checkpoints_path = "C:/Users/cedri/Desktop/Code/ETH/DLProject/Neural_Cellular_Automata_for_diverse_Tree_growing/checkpoints/2024-01-05-00-40-39_VariousTrees_20tree_final_pink/checkpoints"
    # directory = f"{checkpoints_path}/{2600}"
    # config_path = find_file(directory, ".yaml")
    # pretrained_path = find_file(directory, ".pt")

    ct = KeaggleTrainer.from_config(
        "{}/{}/config.yaml".format(base_data_path, dataset),
        config={
            # "pretrained_path": pretrained_path,
            "use_cuda": torch.cuda.is_available(),
            "wandb": False,
            'train_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "train")},
            'val_dataset_config': {'dataset_path': "{}/{}".format(dataset_path, "val")},
        }
    )
    ct.train()
