from src.dataset.combined_dataset import CombinedDataset
from src.dataset.kaeggle_dataset import KaeggleDataset
from src.dataset.nothing_dataset import TestDataset

datasetMappingDict = {"TestDataset": TestDataset, 'KaeggleDataset': KaeggleDataset, 'CombinedDataset': CombinedDataset}
