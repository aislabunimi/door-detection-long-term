from typing import Tuple

from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample, DOOR_LABELS
from doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal
from doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET


class DatasetsCreatorDoorsNoDoorTask:

    def __init__(self, dataset_path: str, folder_name: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()
        self._folder_name = folder_name

    def get_labels(self):
        return DOOR_LABELS

    def create_datasets(self, train_size: float = 0.25, test_size: float = 0.25, random_state: int = 42) -> Tuple[DatasetDoorsFinal, DatasetDoorsFinal]:
        self._dataframe = self._dataframe[self._dataframe.folder_name == self._folder_name]
        self._dataframe = shuffle(self._dataframe, random_state=random_state)

        train_indexes, test_indexes = train_test_split(self._dataframe.index.tolist(), train_size=train_size, test_size=test_size,random_state=random_state)

        train_dataframe, test_dataframe = self._dataframe.loc[train_indexes], self._dataframe.loc[test_indexes]
        train_dataframe = train_dataframe[train_dataframe.label == 1]

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

        return (DatasetDoorsFinal(self._dataset_path, train_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, test_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))

