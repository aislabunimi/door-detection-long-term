import numpy as np
import pandas as pd
import torch
from generic_dataset.dataset_manager import DatasetManager
from generic_dataset.utilities.color import Color
from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample
import doors_detector.utilities.transforms as T
from PIL import Image
from typing import Type, List, Tuple
from doors_detector.dataset.torch_dataset import TorchDataset, SET, TRAIN_SET, TEST_SET


class DatasetDoorsFinal(TorchDataset):
    def __init__(self, dataset_path: str,
                 dataframe: pd.DataFrame,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):

        super(DatasetDoorsFinal, self).__init__(
            dataset_path=dataset_path,
            dataframe=dataframe,
            set_type=set_type,
            std_size=std_size,
            max_size=max_size,
            scales=scales)

        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)

    def load_sample(self, idx) -> Tuple[DoorSample, str, int]:
        row = self._dataframe.iloc[idx]
        folder_name, absolute_count = row.folder_name, row.folder_absolute_count

        door_sample: DoorSample = self._doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        return door_sample, folder_name, absolute_count

