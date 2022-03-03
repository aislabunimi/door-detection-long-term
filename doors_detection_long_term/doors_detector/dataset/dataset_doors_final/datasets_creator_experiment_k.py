import pandas as pd
from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample
from doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal
from doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET


class DatasetsCreatorExperimentK():
    def __init__(self, dataset_path: str, folder_name: str, test_dataframe: pd.DataFrame):
        self._dataset_path = dataset_path
        self._folder_name = folder_name
        self._test_dataframe = test_dataframe
        self._absolute_counts: list = []

    def len_train_set(self):
        return len(self._absolute_counts)

    def add_train_sample(self, absolute_count: int):
        self._absolute_counts.append(absolute_count)

    def create_datasets(self, number_of_samples: int, random_state: int = 42):
        complete_dataframe = DatasetManager(dataset_path=self._dataset_path, sample_class=DoorSample).get_dataframe()
        dataframe_train = complete_dataframe[(complete_dataframe.folder_name == self._folder_name) & (complete_dataframe.folder_absolute_count.isin(self._absolute_counts))]

        if len(dataframe_train.index) > number_of_samples:
            train_indexes, _ = train_test_split(dataframe_train.index, train_size=number_of_samples, random_state=42)
        else:
            train_indexes, _ = train_test_split(dataframe_train.index, train_size=len(dataframe_train.index) - 2, random_state=42)
        dataframe_train = dataframe_train.loc[train_indexes]
        dataframe_test = self._test_dataframe[self._test_dataframe.label == 1]

        return (DatasetDoorsFinal(self._dataset_path, dataframe_train, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, dataframe_test, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))