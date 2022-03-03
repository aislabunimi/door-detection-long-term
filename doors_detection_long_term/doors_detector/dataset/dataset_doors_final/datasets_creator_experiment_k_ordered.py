import pandas as pd
from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample
from doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal
from doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from doors_detector.experiment_k.criterion import CriterionSorting


class DatasetsCreatorExperimentKOrdered:
    def __init__(self, dataset_path: str, folder_name: str, test_dataframe: pd.DataFrame):
        self._dataset_path = dataset_path
        self._folder_name = folder_name
        self._test_dataframe = test_dataframe
        self._absolute_counts: list = []

    def len_train_set(self):
        return len(self._absolute_counts)

    def add_train_sample(self, absolute_count: int, score_image: float):
        self._absolute_counts.append((absolute_count, score_image))

    def create_datasets(self, number_of_samples: int, sorting: CriterionSorting, random_state: int = 42):
        if sorting == CriterionSorting.GROWING:
            self._absolute_counts.sort(key=lambda item: item[1])
        elif sorting == CriterionSorting.DESCENDING:
            self._absolute_counts.sort(key=lambda item: item[1], reverse=True)

        complete_dataframe = DatasetManager(dataset_path=self._dataset_path, sample_class=DoorSample).get_dataframe()
        dataframe_train = complete_dataframe[(complete_dataframe.folder_name == self._folder_name) & (complete_dataframe.folder_absolute_count.isin(map(lambda item: item[0], self._absolute_counts)))]
        dataframe_train = dataframe_train.set_index('folder_absolute_count')
        dataframe_train = dataframe_train.loc[map(lambda item:item[0], self._absolute_counts)]
        dataframe_train.reset_index(level=0, inplace=True)

        if len(dataframe_train.index) > number_of_samples:
            dataframe_train = dataframe_train.head(number_of_samples)

        dataframe_train = shuffle(dataframe_train, random_state=random_state)

        dataframe_test = self._test_dataframe[self._test_dataframe.label == 1]

        return (DatasetDoorsFinal(self._dataset_path, dataframe_train, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, dataframe_test, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))