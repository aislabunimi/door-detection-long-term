from typing import Tuple
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import json
import numpy as np

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.door_sample import DoorSample, DOOR_LABELS
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.doors_dataset import DatasetDoorsIGibson


class DatasetCreatorScenesEpochAnalysis:
    def __init__(self, dataset_path: str):
        self._dataset_path = dataset_path
        self._dataset_manager = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)
        self._dataframe = self._dataset_manager.get_dataframe()
        self._metadata_filename = "metadata.json"

        self._experiment = 1
        self._folder_name = None
        self._use_negatives = False

    def get_labels(self):
        return DOOR_LABELS
    
    def set_experiment_number(self, experiment: int, folder_name: str) -> 'DatasetsCreatorScenes':
        assert experiment == 1 or experiment == 2

        self._experiment = experiment
        self._folder_name = folder_name
        return self
    
    def use_negatives(self, use_negatives: bool) -> 'DatasetsCreatorScenes':
        self._use_negatives = use_negatives
        return self
    
    def create_datasets(self, scene_name:str = None, doors_config:str = None, train_size: float = 0.1, random_state: int = 42) -> Tuple[DatasetDoorsIGibson]:
        def print_information(dataframe):
            information_text = \
            f""">\t Total samples: {len(dataframe.index)}\n""" + \
            f""">\t Samples labels: {sorted(dataframe.label.unique())}\n""" + \
            f""">\t Folders considered: {len(dataframe.folder_name.unique())}"""

            information_text += "\n>\t Folders contributions:"
            for folder_name in sorted(dataframe.folder_name.unique()):
                information_text += f"""\n>\t\t- {folder_name}: {len(dataframe[dataframe.folder_name == folder_name])}"""
                if DoorSample.GET_LABEL_SET():
                    information_text += " of which"
                    for label in sorted(list(DoorSample.GET_LABEL_SET())):
                        information_text += f""" {len(dataframe[(dataframe.folder_name == folder_name) & (dataframe.label == label)])} have label {label}"""

            print(information_text)

        def filter_by_configuration(dataframe_row):
            folder_name = dataframe_row.folder_name
            metadata_filepath = os.path.join(self._dataset_path, folder_name, self._metadata_filename)
            with open(metadata_filepath, "r") as mf:
                metadata = json.loads(mf.read())

            return metadata["doors_method"] == doors_config
        
        def filter_by_scene(dataframe_row):
            folder_name = dataframe_row.folder_name
            metadata_filepath = os.path.join(self._dataset_path, folder_name, self._metadata_filename)
            with open(metadata_filepath, "r") as mf:
                metadata = json.loads(mf.read())

            return metadata["scene"] == scene_name
        
        if isinstance(train_size, float):
            assert train_size in [0.15, 0.25, 0.5, 0.75]

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
        if not scene_name == None:
            scene_mask = shuffled_dataframe.apply(filter_by_scene, axis=1)
            shuffled_dataframe = shuffled_dataframe[scene_mask]
        if doors_config in ["full open", "closed", "random open", "realistic"]:
            config_mask = shuffled_dataframe.apply(filter_by_configuration, axis=1)
            shuffled_dataframe = shuffled_dataframe[config_mask]

        if self._experiment == 1:
            generic_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name != self._folder_name) & (shuffled_dataframe.label == 1)]
            train_index, _ = train_test_split(generic_dataframe.index.tolist(), train_size=0.95, random_state=random_state)

            train_dataframe = generic_dataframe.loc[train_index]

        elif self._experiment == 2:
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
            
        generic_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name != self._folder_name) & (shuffled_dataframe.label == 1)]
        _, validation_index = train_test_split(generic_dataframe.index.tolist(), train_size=0.95, random_state=random_state)
        validation_dataframe = generic_dataframe.loc[validation_index]

        qualified_dataframe = shuffled_dataframe[(shuffled_dataframe.folder_name == self._folder_name) & (shuffled_dataframe.label == 1)]
        [_, _, _, fold_4] = np.array_split(qualified_dataframe.index.to_numpy(), 4)
        test_dataframe = qualified_dataframe.loc[fold_4.tolist()]

        if self._use_negatives:
            test_dataframe = test_dataframe.append(shuffled_dataframe[(shuffled_dataframe.folder_name == self._folder_name) & (shuffled_dataframe.label == 0)], ignore_index = True)

        for title, dataset in zip(["Original frame", "Training frame", "Validation frame", "Test frame"], [self._dataframe, train_dataframe, validation_dataframe, test_dataframe]):
            print(title)
            print_information(dataset)

        return (
            DatasetDoorsIGibson(self._dataset_path, train_dataframe, TRAIN_SET, std_size=512, max_size=800, scales=[512 + i * 32 for i in range(11)]),
            DatasetDoorsIGibson(self._dataset_path, validation_dataframe, TEST_SET, std_size=512, max_size=800, scales=[512 + i * 32 for i in range(11)]),
            DatasetDoorsIGibson(self._dataset_path, test_dataframe, TEST_SET, std_size=512, max_size=800, scales=[512 + i * 32 for i in range(11)])
        )