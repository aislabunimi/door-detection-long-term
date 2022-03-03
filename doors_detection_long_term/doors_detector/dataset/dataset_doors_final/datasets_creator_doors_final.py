from typing import Union, Tuple

import pandas as pd
from sklearn.utils import shuffle

from doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET, SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample, DOOR_LABELS
from sklearn.model_selection import train_test_split

from doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal


class DatasetsCreatorDoorsFinal:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()

        self._experiment = 1
        self._folder_name = None
        self._use_negatives = False

    def get_labels(self):
        return DOOR_LABELS

    def set_experiment_number(self, experiment: int, folder_name: str) -> 'DatasetsCreatorDoorsFinal':
        """
        This method is used to set up the experiment to run.
        1) This first experiment involves training the model using k-1 folders and
        testing it with all the examples in the remaining folder.
        2) The second experiment involves fine-tuning the previously trained model using some examples of the test data used in experiment 1.
        This new training data belongs to a new environment, never seen in the first training phase. The remaining sample of the k-th folder are used as a test set.
        :param experiment: the number of the experiment to perform. It's value must be 0 or 1
        :param folder_name: the name of the folder to use as a test set in experiment 1 or to split into training a test sets in experiment 2.
        :return: the instance of DatasetsCreatorDoorsFinal
        """
        assert experiment == 1 or experiment == 2

        self._experiment = experiment
        self._folder_name = folder_name
        return self

    def use_negatives(self, use_negatives: bool) -> 'DatasetsCreatorDoorsFinal':
        """
        Sets the presence of the negative samples in the test set.
        :param use_negatives: True for including the negatives samples (samples with no doors) in the test set, False to use only positives ones
        :return: the instance of DatasetsDoorsF
        """
        self._use_negatives = use_negatives
        return self

    def create_datasets(self, train_size: float = 0.1, random_state: int = 42) -> Tuple[DatasetDoorsFinal, DatasetDoorsFinal]:
        """
        This method returns the training and test sets.
        :param train_size: the size of the training set in experiment 2. For the first experiment this parameter is not considered, all samples of folders k-1 are considered.
        """
        if isinstance(train_size, float):
            assert 0.0 < train_size < 1.0

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
        if self._experiment == 1:
            train_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name != self._folder_name) & (shuffled_dataframe.label == 1)]
            test_dataframe = shuffled_dataframe[shuffled_dataframe.folder_name == self._folder_name]

            if not self._use_negatives:
                test_dataframe = test_dataframe[test_dataframe.label == 1]

        elif self._experiment == 2:
            shuffled_dataframe = shuffled_dataframe[shuffled_dataframe.folder_name == self._folder_name]
            positive_dataframe = shuffled_dataframe[shuffled_dataframe.label == 1]
            negative_dataframe = shuffled_dataframe[shuffled_dataframe.label == 0]
            train, test = train_test_split(positive_dataframe.index.tolist(), test_size=0.25, random_state=random_state)

            if train_size < 0.75:
                train, _ = train_test_split(train, train_size=train_size * (4 / 3), random_state=random_state)

            train_dataframe = shuffled_dataframe.loc[train]
            test_dataframe = shuffle(pd.concat([shuffled_dataframe.loc[test], negative_dataframe]), random_state=random_state)

            if not self._use_negatives:
                test_dataframe = test_dataframe[test_dataframe.label == 1]

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

        for m, d in zip(['Datasets summary:', 'Train set summary:', 'Test set summary:'], [self._dataframe, train_dataframe, test_dataframe]):
            print(m)
            print_information(d)


        return (DatasetDoorsFinal(self._dataset_path, train_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))