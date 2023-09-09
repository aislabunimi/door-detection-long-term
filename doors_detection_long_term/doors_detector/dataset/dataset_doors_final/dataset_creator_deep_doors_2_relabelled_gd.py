from typing import Union

from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.final_doors_dataset import \
    DatasetDoorsFinalAndDeepDoors2
from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from doors_detection_long_term.doors_detector.dataset.dataset_deep_doors_2_labelled.door_sample import DoorSample as DoorSampleDeepDoors2
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DOOR_LABELS


from doors_detection_long_term.doors_detector.dataset.dataset_deep_doors_2_labelled.dataset_deep_door_2_labelled import DatasetDeepDoors2Labelled


class DatasetsCreatorDeepDoors2LabelledGD:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSampleDeepDoors2)
        self._dataframe = self._dataset_manager.get_dataframe()

    def get_labels(self):
        return DOOR_LABELS

    def creates_dataset(self, fixed_scale=False, random_state: int = 42):

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
        shuffled_dataframe = shuffled_dataframe[shuffled_dataframe.label == 1]

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
            DatasetDoorsFinalAndDeepDoors2('', self._dataset_path, train_dataframe, TRAIN_SET if not fixed_scale else TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
            DatasetDoorsFinalAndDeepDoors2('', self._dataset_path, validation_dataframe, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)])
        )
