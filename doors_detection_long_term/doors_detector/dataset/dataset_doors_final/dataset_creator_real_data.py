from typing import Union

import numpy as np
import pandas as pd
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DOOR_LABELS
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample


class DatasetsCreatorRealData:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()

    def get_labels(self):
        return DOOR_LABELS

    def create_datasets(self, folder_name: str, train_size: float, random_state: int = 42):

        self._dataframe = self._dataframe[(self._dataframe.label == 1) & (self._dataframe.folder_name == folder_name)]
        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)

        [fold_1, fold_2, fold_3, fold_4] = np.array_split(shuffled_dataframe.index.to_numpy(), 4)

        if train_size == 0.25:
            train_dataframe = shuffled_dataframe.loc[fold_1.tolist()]
        elif train_size == 0.50:
            train_dataframe = shuffled_dataframe.loc[fold_1.tolist() + fold_2.tolist()]
        elif train_size == 0.75:
            train_dataframe = shuffled_dataframe.loc[fold_1.tolist() + fold_2.tolist() + fold_3.tolist()]
        else:
            raise Exception('Train size must be 0.25, 0.50 or 0.75')

        test_dataframe = shuffled_dataframe.loc[fold_4.tolist()]

        def print_information(dataframe):
            print(f'    - total samples = {len(dataframe.index)}\n'
                  f'    - Folders considered: {sorted(dataframe.folder_name.unique())}\n'
                  f'    - Labels considered: {sorted(dataframe.label.unique())}\n'
                  )
            print()

        for m, d in zip(['Datasets summary:', 'Train summary', 'Test summary:'], [self._dataframe, train_dataframe, test_dataframe]):
            print(m)
            print_information(d)

        return (
            DatasetDoorsFinal(self._dataset_path, train_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
            DatasetDoorsFinal(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[480 + i * 32 for i in range(11)])
        )
