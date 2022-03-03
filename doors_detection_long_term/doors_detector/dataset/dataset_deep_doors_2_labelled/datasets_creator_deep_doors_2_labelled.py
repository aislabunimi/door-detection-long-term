from typing import Union
from doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from deep_doors_2.door_sample import DoorSample, DOOR_LABELS
from sklearn.model_selection import train_test_split

from doors_detector.dataset.dataset_deep_doors_2_labelled.dataset_deep_door_2_labelled import DatasetDeepDoors2Labelled


class DatasetsCreatorDeepDoors2Labelled:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()

    def get_labels(self):
        return DOOR_LABELS

    def consider_samples_with_label(self, label: int) -> 'DatasetsCreatorGibson':
        """
        This method sets the class to consider only the samples with the given label.
        Other samples (with different labels) are not considered in the datasets' creations
        :param label: the label of the samples to include in the dataset
        :return: DatasetsCreator itself
        """
        self._dataframe = self._dataframe[self._dataframe.label == label]
        return self

    def creates_dataset(self, train_size: Union[float, int],
                        test_size: Union[float, int],
                        random_state: int = 42):
        """
        This method returns the training and test sets.
        :param train_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
                            If int, represents the absolute number of train samples.
        :param test_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
                            If int, represents the absolute number of test samples.
        :return:
        """
        if isinstance(train_size, float):
            assert 0.0 <= train_size <= 1.0

        if isinstance(test_size, float):
            assert 0.0 <= test_size <= 1.0

        assert isinstance(test_size, float) and isinstance(train_size, float) or isinstance(test_size, int) and isinstance(train_size, int)


        train, test = train_test_split(self._dataframe.index.tolist(), train_size=train_size, test_size=test_size, random_state=random_state)
        train_dataframe = self._dataframe.loc[train]
        test_dataframe = self._dataframe.loc[test]

        def print_information(dataframe):
            print(f'    - total samples = {len(dataframe.index)}\n'
                  f'    - Folders considered: {sorted(dataframe.folder_name.unique())}\n'
                  f'    - Labels considered: {sorted(dataframe.label.unique())}\n'
                  )
            print()

        for m, d in zip(['Datasets summary:', 'Train set summary:', 'Test set summary:'], [self._dataframe, train_dataframe, test_dataframe]):
            print(m)
            print_information(d)

        return (
            DatasetDeepDoors2Labelled(self._dataset_path, train_dataframe, TRAIN_SET, std_size=480, max_size=1000, scales=[480 + i * 32 for i in range(11)]),
            DatasetDeepDoors2Labelled(self._dataset_path, test_dataframe, TEST_SET, std_size=480, max_size=1000, scales=[480 + i * 32 for i in range(11)])
        )
