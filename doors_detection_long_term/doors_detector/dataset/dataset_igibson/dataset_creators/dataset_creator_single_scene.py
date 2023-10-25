from typing import Tuple
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
import json

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TRAIN_SET, TEST_SET
from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.door_sample import DoorSample, DOOR_LABELS
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.doors_dataset import DatasetDoorsIGibson


class DatasetCreatorSingleScene:
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
    
    def use_negatives(self, use_negatives: bool) -> 'DatasetsCreatorSingleScene':
        self._use_negatives = use_negatives
        return self
    
    def create_datasets(self, scene_name:str, doors_config:str = None, random_state: int = 42) -> Tuple[DatasetDoorsIGibson]:
        ## rows with doors
        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)
        shuffled_dataframe = shuffled_dataframe[shuffled_dataframe.label == 1]

        ## rows from scene_name
        scene_mask = shuffled_dataframe.apply(self.filter_by_scene, axis=1, scene_name=scene_name)
        shuffled_dataframe = shuffled_dataframe[scene_mask]

        ## rows with doors_config
        if doors_config in ["full open", "closed", "random open", "realistic"]:
            config_mask = shuffled_dataframe.apply(self.filter_by_configuration, axis=1, doors_config=doors_config)
            shuffled_dataframe = shuffled_dataframe[config_mask]

        ## Splitting
        train_index, validation_index = train_test_split(shuffled_dataframe.index.tolist(), train_size=0.95, random_state=random_state)
        train_dataframe = shuffled_dataframe.loc[train_index]
        validation_dataframe = shuffled_dataframe.loc[validation_index]

        ## printing infos
        for title, dataset in zip(["Original frame", "Training frame", "Validation frame"], [self._dataframe, train_dataframe, validation_dataframe]):
            print(title)
            self.print_informations(dataset)

        return (
            DatasetDoorsIGibson(self._dataset_path, train_dataframe, TRAIN_SET, std_size=512, max_size=800, scales=[512 + i * 32 for i in range(11)]),
            DatasetDoorsIGibson(self._dataset_path, validation_dataframe, TEST_SET, std_size=512, max_size=800, scales=[512 + i * 32 for i in range(11)])
        )
    
    def print_informations(self, dataframe):
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

    def filter_by_configuration(self, dataframe_row, doors_config):
        """
        Filters out images obtained in run with a configuration not equal to doors_config
        """
        folder_name = dataframe_row.folder_name
        metadata_filepath = os.path.join(self._dataset_path, folder_name, self._metadata_filename)
        with open(metadata_filepath, "r") as mf:
            metadata = json.loads(mf.read())

        return metadata["doors_method"] == doors_config
    
    def filter_by_scene(self, dataframe_row, scene_name):
        """
        Filters out images obtained from a scene with a different name than scene_name
        """
        folder_name = dataframe_row.folder_name
        metadata_filepath = os.path.join(self._dataset_path, folder_name, self._metadata_filename)
        with open(metadata_filepath, "r") as mf:
            metadata = json.loads(mf.read())

        return metadata["scene"] == scene_name