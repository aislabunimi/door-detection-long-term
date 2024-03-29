from typing import Union, Tuple, List

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET, SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample, DOOR_LABELS
from sklearn.model_selection import train_test_split

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal


class DatasetsCreatorDoorsFinalEpochAnalysis:
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
        1) This first experiment consists on training a general model on a set of environment E \ {e}
            the model is validated using a subset of the examples collected in E \ {e}
            and it is tested using the 25% of the examples collected in e
        2) The second experiment consists of fine-tune the previously trained model using some examples collected in e.
            It is trained using the 25%, 50% and 75% of the examples collected in e and tested with the remaining 25% of images (that are the same in the experiment1)
            It is validated using the previous validation set to understand the model forgetting.
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

    def create_datasets(self, train_size: float = 0.1, random_state: int = 42) -> List[DatasetDoorsFinal]:
        """
        This method returns the training and test sets.
        :param train_size: the size of the training set in experiment 2. For the first experiment this parameter is not considered, all samples of folders k-1 are considered.
        """
        if isinstance(train_size, float):
            assert train_size == 0.15 or train_size == 0.25 or train_size == 0.5 or train_size == 0.75

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)

        if self._experiment == 1:
            # Extract the dataframe for training the generic model (with the examples acquired in the set of environment E \ {e}

            # Build train and validation set
            generic_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name != self._folder_name) & (shuffled_dataframe.label == 1)]
            train_index, _ = train_test_split(generic_dataframe.index.tolist(), train_size=0.95, random_state=random_state)

            train_dataframe = generic_dataframe.loc[train_index]


        elif self._experiment == 2:

            # Extract the dataframe for testing in the environment e (not used during the training)
            qualified_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name == self._folder_name) & (shuffled_dataframe.label == 1)]
            [fold_1, fold_2, fold_3, _] = np.array_split(qualified_dataframe.index.to_numpy(), 4)

            if train_size <= 0.25:
                fold_1 = fold_1.tolist()
                train_dataframe = qualified_dataframe.loc[fold_1[:int((train_size / 0.25) * len(fold_1))]]

            elif train_size == 0.50:
                train_dataframe = qualified_dataframe.loc[fold_1.tolist() + fold_2.tolist()]
            elif train_size == 0.75:
                train_dataframe = qualified_dataframe.loc[fold_1.tolist() + fold_2.tolist() + fold_3.tolist()]
            else:
                raise Exception('Train size must be 0.25, 0.50 or 0.75')

        # Build validation set
        generic_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name != self._folder_name) & (shuffled_dataframe.label == 1)]
        _, validation_index = train_test_split(generic_dataframe.index.tolist(), train_size=0.95, random_state=random_state)
        validation_dataframe = generic_dataframe.loc[validation_index]

        # Build test set
        qualified_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name == self._folder_name) & (shuffled_dataframe.label == 1)]
        [_, _, _, fold_4] = np.array_split(qualified_dataframe.index.to_numpy(), 4)

        test_dataframe = qualified_dataframe.loc[fold_4.tolist()]

        if self._use_negatives:
            test_dataframe = test_dataframe.append(shuffled_dataframe[(shuffled_dataframe.folder_name == self._folder_name) & (shuffled_dataframe.label == 0)], ignore_index = True)

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

        for m, d in zip(['Datasets summary:', 'Train set summary:', 'Validation set summary', 'Test set summary:'], [self._dataframe, train_dataframe, validation_dataframe, test_dataframe]):
            print(m)
            print_information(d)


        return (DatasetDoorsFinal(self._dataset_path, train_dataframe, TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, validation_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))