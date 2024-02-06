from typing import Tuple
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import json
import pandas as pd
import math

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.door_sample import DoorSample as iGibsonDoorSample, DOOR_LABELS
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample as GibsonDoorSample
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.doors_dataset import DatasetDoorsIGibsonGibson


class DatasetCreatorMixed:
    def __init__(self, igibson_dataset_path:str, gibson_dataset_path:str):
        ## iGibson
        self._igibson_dataset_path = igibson_dataset_path
        self._igibson_dataset_manager = DatasetManager(dataset_path=igibson_dataset_path, sample_class=iGibsonDoorSample)
        self._igibson_dataframe = self._igibson_dataset_manager.get_dataframe()
        self._metadata_filename = "metadata.json"

        ## Gibson
        self._gibson_dataset_path = gibson_dataset_path
        self._gibson_dataset_manager = DatasetManager(dataset_path=gibson_dataset_path, sample_class=GibsonDoorSample)
        self._gibson_dataframe = self._gibson_dataset_manager.get_dataframe()

        self._experiment = 1
        self._folder_name = None
        self._use_negatives = False

    def get_labels(self):
        return DOOR_LABELS
    
    def use_negatives(self, use_negatives: bool) -> 'DatasetsCreatorMixed':
        self._use_negatives = use_negatives
        return self
    
    def create_datasets(self, split:int = 0.5, max_samples:int = 10000, doors_config:str = None, random_state: int = 42) -> Tuple[DatasetDoorsIGibsonGibson]:
        ## rows with doors
        igibson_dataframe = self._igibson_dataframe[self._igibson_dataframe['label'] == 1] if not self._use_negatives else self._igibson_dataframe
        gibson_dataframe = self._gibson_dataframe[self._gibson_dataframe['label'] == 1] if not self._use_negatives else self._gibson_dataframe

        ## igibson rows with doors config
        if doors_config in ["full open", "closed", "random open", "realistic"]:
            config_mask = igibson_dataframe.apply(self.filter_by_configuration, axis=1, doors_config=doors_config)
            igibson_dataframe = igibson_dataframe[config_mask]

        ## splitted sampling
        igibson_number = math.floor(split * max_samples)
        gibson_number = max_samples - igibson_number
        igibson_dataframe = igibson_dataframe.sample(igibson_number, random_state=random_state)
        gibson_dataframe = gibson_dataframe.sample(gibson_number, random_state=random_state)

        ## Adding dataset name
        igibson_dataframe['dataset'] = ['igibson' for _ in range(len(igibson_dataframe))]
        gibson_dataframe['dataset'] = ['gibson' for _ in range(len(gibson_dataframe))]

        ## joining dataframes and shuffling
        self._mixed_dataframe = pd.concat([igibson_dataframe, gibson_dataframe], ignore_index=True)
        shuffled_dataframe = shuffle(self._mixed_dataframe, random_state=random_state)

        ## Splitting
        train_index, validation_index = train_test_split(shuffled_dataframe.index.tolist(), train_size=0.95, random_state=random_state)
        train_dataframe = shuffled_dataframe.loc[train_index]
        validation_dataframe = shuffled_dataframe.loc[validation_index]

        ## printing infos
        for title, dataset in zip(["Original frame", "Training frame", "Validation frame"], [self._mixed_dataframe, train_dataframe, validation_dataframe]):
            print(title)
            self.print_information(dataset)

        return (
            DatasetDoorsIGibsonGibson(self._igibson_dataset_path, self._gibson_dataset_path, train_dataframe, TRAIN_SET, std_size=512, max_size=800, scales=[256 + i * 32 for i in range(11)]),
            DatasetDoorsIGibsonGibson(self._igibson_dataset_path, self._gibson_dataset_path, validation_dataframe, TEST_SET, std_size=512, max_size=800, scales=[256 + i * 32 for i in range(11)])
        )
    
    def print_information(self, dataframe):
        information_text = \
        f""">\t Total samples: {len(dataframe.index)}\n""" + \
        f""">\t Samples labels: {sorted(dataframe.label.unique())}\n""" + \
        f""">\t Folders considered: {len(dataframe.folder_name.unique())}"""

        information_text += "\n>\t Folders contributions:"
        for folder_name in sorted(dataframe.folder_name.unique()):
            information_text += f"""\n>\t\t- {folder_name}: {len(dataframe[dataframe.folder_name == folder_name])}"""
            if iGibsonDoorSample.GET_LABEL_SET():
                information_text += " of which"
                for label in sorted(list(iGibsonDoorSample.GET_LABEL_SET())):
                    information_text += f""" {len(dataframe[(dataframe.folder_name == folder_name) & (dataframe.label == label)])} have label {label}"""

        print(information_text)

    def filter_by_configuration(self, dataframe_row, doors_config):
        folder_name = dataframe_row.folder_name
        metadata_filepath = os.path.join(self._igibson_dataset_path, folder_name, self._metadata_filename)
        with open(metadata_filepath, "r") as mf:
            metadata = json.loads(mf.read())

        return metadata["doors_method"] == doors_config