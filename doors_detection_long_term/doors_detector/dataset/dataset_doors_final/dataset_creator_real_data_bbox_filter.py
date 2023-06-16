from typing import Union

import numpy as np
import pandas as pd
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinalRealData
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample_real_data import DOOR_LABELS
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample_real_data import DoorSample


class DatasetsCreatorRealDataBboxFilter:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()

    def get_labels(self):
        return DOOR_LABELS

    def create_datasets(self, folder_name: str, train_size: float, control_size: float, fine_tune_size: float, test_size: float, random_state: int = 42):
        assert train_size + control_size + test_size + fine_tune_size <= 1.0 and train_size > .0 and test_size > .0 and control_size > .0 and fine_tune_size > .0

        self._dataframe = self._dataframe[(self._dataframe.label == 1) & (self._dataframe.folder_name == folder_name)]
        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)

        indexes = shuffled_dataframe.index.to_numpy().tolist()

        train_size_int = int(len(indexes) * train_size)
        control_size_int = train_size_int + int(len(indexes) * control_size)
        fine_tune_size_int = control_size_int + int(len(indexes) * fine_tune_size)
        test_size_int = int(len(indexes) * test_size)

        # In case of wrong rounded integers
        fine_tune_size_int += len(indexes) - (train_size_int + control_size_int + fine_tune_size_int + test_size_int)

        train_dataframe = shuffled_dataframe.loc[indexes[:train_size_int]]
        control_dataframe = shuffled_dataframe.loc[indexes[train_size_int:control_size_int]]
        fine_tune_dataframe = shuffled_dataframe.loc[indexes[control_size_int:fine_tune_size_int]]


        test_dataframe = shuffled_dataframe.loc[indexes[:-test_size_int]]

        def print_information(dataframe):
            print(f'    - total samples = {len(dataframe.index)}\n'
                  f'    - Folders considered: {sorted(dataframe.folder_name.unique())}\n'
                  f'    - Labels considered: {sorted(dataframe.label.unique())}\n'
                  )
            print()

        for m, d in zip(['Train summary', 'Control summary', 'Fine tune simmary', 'Test summary:'], [self._dataframe, train_dataframe, control_dataframe, fine_tune_dataframe, test_dataframe]):
            print(m)
            print_information(d)

        return (
            DatasetDoorsFinalRealData(self._dataset_path, train_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
            DatasetDoorsFinalRealData(self._dataset_path, control_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[480 + i * 32 for i in range(11)]),
            DatasetDoorsFinalRealData(self._dataset_path, fine_tune_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[480 + i * 32 for i in range(11)]),
            DatasetDoorsFinalRealData(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[480 + i * 32 for i in range(11)])
        )
