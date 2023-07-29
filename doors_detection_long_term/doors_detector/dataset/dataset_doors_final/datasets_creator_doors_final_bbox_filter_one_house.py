from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET, SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample, DOOR_LABELS
from sklearn.model_selection import train_test_split

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal


class DatasetsCreatorDoorsFinalBBoxFilterOneHouse:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()
        self._use_negatives = False

    def get_labels(self):
        return DOOR_LABELS

    def create_datasets(self, folder_name: str, random_state: int = 42) -> Tuple[DatasetDoorsFinal, DatasetDoorsFinal]:
        """
        Returns D, where D = De(75) U de(25)
        """

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
        qualified_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name == folder_name) & (shuffled_dataframe.label == 1)]
        [fold_1, fold_2, fold_3, fold_4] = np.array_split(qualified_dataframe.index.to_numpy(), 4)

        train_dataframe = qualified_dataframe.loc[fold_1.tolist() + fold_2.tolist() + fold_3.tolist()]
        test_dataframe = qualified_dataframe.loc[fold_4.tolist()]

        def print_information(dataframe):
            print(f'    - total samples = {len(dataframe.index)}\n'
                  f'    - Folders considered: {sorted(dataframe.folder_name.unique())}\n'
                  f'    - Labels considered: {sorted(dataframe.label.unique())}\n'
                  f'    - Total samples in folder: ')
            for folder in sorted(dataframe.folder_name.unique()):
                print(f'        - {folder}: {len(dataframe[dataframe.folder_name == folder])} samples')
                if DoorSample.GET_LABEL_SET():
                    print(f'        Samples per label:')
                    for label in sorted(list(DoorSample.GET_LABEL_SET())):
                        print(f'            - {label}: {len(dataframe[(dataframe.folder_name == folder) & (dataframe.label == label)])}')
            print()

        for m, d in zip(['Datasets summary:', 'Train student set summary:', 'Validation set summary (generic)', 'unlabelled images set', 'Test set summary:'], [self._dataframe, train_dataframe, test_dataframe]):
            print(m)
            print_information(d)

        return (DatasetDoorsFinal(self._dataset_path, train_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))