from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET, SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample, DOOR_LABELS
from sklearn.model_selection import train_test_split

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal


class DatasetsCreatorAllEnvs:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()

        self._experiment = 1
        self._folder_name = None
        self._use_negatives = False

    def get_labels(self):
        return DOOR_LABELS

    def use_negatives(self, use_negatives: bool) -> 'DatasetsCreatorDoorsFinal':
        """
        Sets the presence of the negative samples in the test set.
        :param use_negatives: True for including the negatives samples (samples with no doors) in the test set, False to use only positives ones
        :return: the instance of DatasetsDoorsF
        """
        self._use_negatives = use_negatives
        return self

    def create_datasets(self, random_state: int = 42) -> Tuple[DatasetDoorsFinal]:
        """
        This method returns the training and test sets.
        :param train_size: the size of the training set in experiment 2. For the first experiment this parameter is not considered, all samples of folders k-1 are considered.
        """

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
        shuffled_dataframe = shuffled_dataframe[shuffled_dataframe.label == 1]

        train_index, validation_index = train_test_split(shuffled_dataframe.index.tolist(), train_size=0.95, random_state=random_state)

        train_dataframe = shuffled_dataframe.loc[train_index]
        validation_dataframe = shuffled_dataframe.loc[validation_index]

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

        for m, d in zip(['Datasets summary:', 'Train set summary:', 'Validation set summary'], [self._dataframe, train_dataframe, validation_dataframe]):
            print(m)
            print_information(d)


        return (DatasetDoorsFinal(self._dataset_path, train_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, validation_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))