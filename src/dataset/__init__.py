from src.dataset.combined_dataset import CombinedDataset
from src.dataset.kaeggle_dataset import KaeggleDataset
from src.dataset.nothing_dataset import NothingDataset

datasetMappingDict = {"NothingDataset": NothingDataset, 'KaeggleDataset': KaeggleDataset, 'CombinedDataset': CombinedDataset}
