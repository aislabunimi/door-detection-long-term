import pandas as pd
from deep_doors_2.door_sample import DoorSample
from generic_dataset.dataset_manager import DatasetManager
from typing import List, Tuple
from doors_detector.dataset.torch_dataset import TorchDataset, SET


class DatasetDeepDoors2Labelled(TorchDataset):
    def __init__(self, dataset_path: str,
                 dataframe: pd.DataFrame,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):

        super(DatasetDeepDoors2Labelled, self).__init__(
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

