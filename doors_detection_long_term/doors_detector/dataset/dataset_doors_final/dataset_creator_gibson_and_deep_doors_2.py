from typing import Union

import pandas as pd
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import \
    DatasetDoorsFinalAndDeepDoors2
from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from doors_detection_long_term.doors_detector.dataset.dataset_deep_doors_2_labelled.door_sample import DoorSample as DoorSampleDeepDoors2
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DOOR_LABELS
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample as DoorSampleGibson


from doors_detection_long_term.doors_detector.dataset.dataset_deep_doors_2_labelled.dataset_deep_door_2_labelled import DatasetDeepDoors2Labelled


class DatasetsCreatorGibsonAndDeepDoors2:
    def __init__(self, dataset_path_gibson: str, dataset_path_deep_doors_2: str):
        self._dataset_path_gibson = dataset_path_gibson
        self._dataset_path_deep_doors_2 = dataset_path_deep_doors_2
        self._dataset_manager_gibson = DatasetManager(dataset_path=dataset_path_gibson, sample_class=DoorSampleGibson)
        self._dataset_manager_deep_doors_2 = DatasetManager(dataset_path=dataset_path_deep_doors_2, sample_class=DoorSampleDeepDoors2)
        self._dataframe_gibson = self._dataset_manager_gibson.get_dataframe()
        self._dataframe_deep_doors_2 = self._dataset_manager_deep_doors_2.get_dataframe()
        self._dataframe = pd.concat([self._dataframe_gibson, self._dataframe_deep_doors_2], ignore_index=True)

    def get_labels(self):
        return DOOR_LABELS

    def creates_dataset(self, half: bool, random_state: int = 42):

        self._dataframe = self._dataframe[self._dataframe.label == 1]
        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)

        if half:
            shuffled_dataframe = shuffled_dataframe[: int(len(shuffled_dataframe.index) / 2)]

        train, validation = train_test_split(shuffled_dataframe.index.tolist(), train_size=0.95, random_state=random_state)
        train_dataframe = self._dataframe.loc[train]
        validation_dataframe = self._dataframe.loc[validation]

        def print_information(dataframe):
            print(f'    - total samples = {len(dataframe.index)}\n'
                  f'    - Folders considered: {sorted(dataframe.folder_name.unique())}\n'
                  f'    - Labels considered: {sorted(dataframe.label.unique())}\n'
                  )
            print()

        for m, d in zip(['Datasets summary:', 'Train summary', 'Validation summary:'], [self._dataframe, train_dataframe, validation_dataframe]):
            print(m)
            print_information(d)

        return (
            DatasetDoorsFinalAndDeepDoors2(self._dataset_path_gibson, self._dataset_path_deep_doors_2, train_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
            DatasetDoorsFinalAndDeepDoors2(self._dataset_path_gibson, self._dataset_path_deep_doors_2, validation_dataframe, TEST_SET, std_size=256, max_size=800, scales=[480 + i * 32 for i in range(11)])
        )
