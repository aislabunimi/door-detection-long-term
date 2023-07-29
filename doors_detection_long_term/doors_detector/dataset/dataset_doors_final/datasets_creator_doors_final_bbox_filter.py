from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET, SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample, DOOR_LABELS
from sklearn.model_selection import train_test_split

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal


class DatasetsCreatorDoorsFinalBBoxFilter:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()
        self._use_negatives = False

    def get_labels(self):
        return DOOR_LABELS

    def use_negatives(self, use_negatives: bool) -> 'DatasetsCreatorDoorsFinalBBoxFilter':

        self._use_negatives = use_negatives
        return self

    def create_datasets(self, folder_name: str, train_size_student: float, random_state: int = 42) -> Tuple[DatasetDoorsFinal, DatasetDoorsFinal, DatasetDoorsFinal, DatasetDoorsFinal]:
        """
        The method returns the following datasets:
        - train_student: the dataset used to train the student. For the general detectors are all the examples in the other environments
                        in case of a qualified detectors, it contains the examples used for fine-tuning it
        - validation_student: the dataset to validate the model (it contains only a few examples from the dataset of other environments)
        - unlabelled_bbox_filter: contains all the example from the target evironments that are not been used to train the qualified detectors
                                    all the 75% in case of the GD, otherwise the difference between the 75% and the train student_dataset
        - test_set: contains the last 25% examples of the target environment, always used for testing
        """
        if isinstance(train_size_student, float):
            assert train_size_student == .0 or train_size_student == 0.15 or train_size_student == 0.25 or train_size_student == 0.5 or train_size_student == 0.75

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
        generic_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name != folder_name) & (shuffled_dataframe.label == 1)]
        qualified_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name == folder_name) & (shuffled_dataframe.label == 1)]
        [fold_1, fold_2, fold_3, fold_4] = np.array_split(qualified_dataframe.index.to_numpy(), 4)
        train_index_student, validation_index_student = train_test_split(generic_dataframe.index.tolist(), train_size=0.95, random_state=random_state)
        validation_dataframe_student = generic_dataframe.loc[validation_index_student]

        if train_size_student == .0:
            train_dataframe_student = generic_dataframe.loc[train_index_student]
            unlabelled_dataframe_bbox_filter = qualified_dataframe.loc[fold_1.tolist() + fold_2.tolist() + fold_3.tolist()]
        elif train_size_student <= 0.25:
            fold_1 = fold_1.tolist()
            train_dataframe_student = qualified_dataframe.loc[fold_1[:int((train_size_student / 0.25) * len(fold_1))]]
            unlabelled_dataframe_bbox_filter = qualified_dataframe.loc[fold_1[int((train_size_student / 0.25) * len(fold_1)):] + fold_2.tolist() + fold_3.tolist()]
        elif train_size_student == 0.50:
            train_dataframe_student = qualified_dataframe.loc[fold_1.tolist() + fold_2.tolist()]
            unlabelled_dataframe_bbox_filter = qualified_dataframe.loc[fold_3.tolist()]
        elif train_size_student == 0.75:
            train_dataframe_student = qualified_dataframe.loc[fold_1.tolist() + fold_2.tolist() + fold_3.tolist()]
            unlabelled_dataframe_bbox_filter = qualified_dataframe.loc[[]]
        else:
            raise Exception('Train size must be 0.25, 0.50 or 0.75')

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

        for m, d in zip(['Datasets summary:', 'Train student set summary:', 'Validation set summary (generic)', 'unlabelled images set', 'Test set summary:'], [self._dataframe, train_dataframe_student, validation_dataframe_student, unlabelled_dataframe_bbox_filter, test_dataframe]):
            print(m)
            print_information(d)

        return (DatasetDoorsFinal(self._dataset_path, train_dataframe_student, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, validation_dataframe_student, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, unlabelled_dataframe_bbox_filter, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))