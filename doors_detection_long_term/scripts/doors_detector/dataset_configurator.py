import numpy as np
import math
from random import shuffle
from doors_detection_long_term.doors_detector.dataset.dataset_deep_doors_2_labelled.datasets_creator_deep_doors_2_labelled import DatasetsCreatorDeepDoors2Labelled
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.dataset_creator_all_envs import \
    DatasetsCreatorAllEnvs
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.dataset_creator_deep_doors_2_relabelled_gd import \
    DatasetsCreatorDeepDoors2LabelledGD
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.dataset_creator_gibson_and_deep_doors_2 import \
    DatasetsCreatorGibsonAndDeepDoors2
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.dataset_creator_real_data import \
    DatasetsCreatorRealData
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.dataset_creator_real_data_bbox_filter import \
    DatasetsCreatorRealDataBboxFilter
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final_bbox_filter import \
    DatasetsCreatorDoorsFinalBBoxFilter
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final_epoch_analysis import \
    DatasetsCreatorDoorsFinalEpochAnalysis
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_no_door_task import \
    DatasetsCreatorDoorsNoDoorTask
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.dataset_creators.dataset_creator_all_scenes import \
    DatasetCreatorAllScenes as IGibsonDatasetCreatorAllScenes
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.dataset_creators.dataset_creator_single_scene import \
    DatasetCreatorSingleScene as IGibsonDatasetCreatorSingleScene
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.dataset_creators.dataset_creator_epoch_analysis import \
    DatasetCreatorScenesEpochAnalysis as IGibsonDatasetCreatorScenesEpochAnalysis
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.dataset_creators.dataset_creator_mixed import \
    DatasetCreatorMixed as IGibsonGibsonDatasetCreator

# The path in which the trained model are saved and loaded
# If the string is empty, they are saved in a folder in this repository (/models/train_params/)
trained_models_path = "/home/aislab/trained_models"

deep_doors_2_labelled_dataset_path = '/home/aislab/deep_doors_2_labelled'
final_doors_dataset_path = '/home/aislab/gibson_dataset/'
real_final_doors_dataset_path = '/home/aislab/final_doors_dataset_real'
igibson_doors_dataset_path = '/home/aislab/igibson_dataset'


def get_deep_doors_2_labelled_sets():
    dataset_creator = DatasetsCreatorDeepDoors2Labelled(dataset_path=deep_doors_2_labelled_dataset_path)
    dataset_creator.consider_samples_with_label(label=1)
    train, test = dataset_creator.creates_dataset(train_size=0.8, test_size=0.2)
    labels = dataset_creator.get_labels()

    return train, test, labels, {0: (1, 0, 0), 1: (0, 0, 1), 2: (0, 1, 0)}


def get_final_doors_dataset(experiment: int, folder_name: str, train_size: float = 0.1, use_negatives: bool = False):
    dataset_creator = DatasetsCreatorDoorsFinal(dataset_path=final_doors_dataset_path)
    dataset_creator.set_experiment_number(experiment=experiment, folder_name=folder_name)
    dataset_creator.use_negatives(use_negatives=use_negatives)
    train, test = dataset_creator.create_datasets(train_size=train_size)
    labels = dataset_creator.get_labels()

    return train, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)


def get_final_doors_dataset_epoch_analysis(experiment: int, folder_name: str, train_size: float = 0.1, use_negatives: bool = False):
    dataset_creator = DatasetsCreatorDoorsFinalEpochAnalysis(dataset_path=final_doors_dataset_path)
    dataset_creator.set_experiment_number(experiment=experiment, folder_name=folder_name)
    dataset_creator.use_negatives(use_negatives=use_negatives)
    train, validation, test = dataset_creator.create_datasets(train_size=train_size)
    labels = dataset_creator.get_labels()

    return train, validation, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)


def get_final_doors_dataset_door_no_door_task(folder_name: str, train_size: float = 0.25, test_size: float = 0.25):
    dataset_creator = DatasetsCreatorDoorsNoDoorTask(dataset_path=final_doors_dataset_path, folder_name=folder_name)
    train, test = dataset_creator.create_datasets(train_size=train_size, test_size=test_size)
    labels = dataset_creator.get_labels()

    return train, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_all_envs():
    dataset_creator = DatasetsCreatorAllEnvs(dataset_path=final_doors_dataset_path)
    train, validation = dataset_creator.create_datasets()
    labels = dataset_creator.get_labels()

    return train, validation, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_deep_doors_2_relabelled_dataset_for_gd():
    dataset_creator = DatasetsCreatorDeepDoors2LabelledGD(dataset_path=deep_doors_2_labelled_dataset_path)

    train, validation = dataset_creator.creates_dataset()
    labels = dataset_creator.get_labels()

    return train, validation, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_gibson_and_deep_door_2_dataset(half: bool):
    dataset_creator = DatasetsCreatorGibsonAndDeepDoors2(dataset_path_gibson=final_doors_dataset_path, dataset_path_deep_doors_2=deep_doors_2_labelled_dataset_path)

    train, validation = dataset_creator.creates_dataset(half=half)
    labels = dataset_creator.get_labels()

    return train, validation, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_real_data(folder_name: str, train_size: float = 0.1):
    dataset_creator = DatasetsCreatorRealData(dataset_path=real_final_doors_dataset_path)
    train, test = dataset_creator.create_datasets(folder_name=folder_name, train_size=train_size, )
    labels = dataset_creator.get_labels()

    return train, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_bbox_filter(folder_name: str, train_size_student: float = .0):
    dataset_creator = DatasetsCreatorDoorsFinalBBoxFilter(dataset_path=final_doors_dataset_path)
    train_student, validation_student, unlabelled_bbox_filter, test = dataset_creator.create_datasets(folder_name=folder_name, train_size_student=train_size_student)
    labels = dataset_creator.get_labels()

    return train_student, validation_student, unlabelled_bbox_filter, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_real_data_bbox_filter(folder_name: str, train_size: float = 0.15, control_size: float = 0.15, fine_tune_size: float = 0.45, test_size: float = 0.25):
    dataset_creator = DatasetsCreatorRealDataBboxFilter(dataset_path=real_final_doors_dataset_path)
    train, control, fine_tune, test = dataset_creator.create_datasets(folder_name=folder_name, control_size=control_size, fine_tune_size=fine_tune_size, train_size=train_size, test_size=test_size )
    labels = dataset_creator.get_labels()

    return train, control, fine_tune, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

## iGibson datasets

def get_igibson_dataset_all_scenes(doors_config:str = None, step:int = None):
    dataset_creator = IGibsonDatasetCreatorAllScenes(dataset_path=igibson_doors_dataset_path)
    train_set, validation_set = dataset_creator.create_datasets(doors_config, step)
    labels = dataset_creator.get_labels()

    return train_set, validation_set, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_igibson_dataset_scene(scene_name:str, doors_config:str = None):
    dataset_creator = IGibsonDatasetCreatorSingleScene(dataset_path=igibson_doors_dataset_path)
    train_set, validation_set = dataset_creator.create_datasets(scene_name, doors_config)
    labels = dataset_creator.get_labels()

    return train_set, validation_set, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_igibson_dataset_epoch_analysis(experiment: int, folder_name: str, scene_name:str = None, doors_config:str = None, train_size: float = 0.1, use_negatives: bool = False):
    dataset_creator = IGibsonDatasetCreatorScenesEpochAnalysis(dataset_path=igibson_doors_dataset_path)
    dataset_creator.set_experiment_number(experiment=experiment, folder_name=folder_name)
    dataset_creator.use_negatives(use_negatives=use_negatives)
    train, validation, test = dataset_creator.create_datasets(scene_name, doors_config, train_size=train_size)
    labels = dataset_creator.get_labels()

    return train, validation, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_mixed_gibson_igibson_dataset(split:float = 0.5, max_samples:int = 10000, doors_config:str = None):
    dataset_creator = IGibsonGibsonDatasetCreator(igibson_dataset_path=igibson_doors_dataset_path, gibson_dataset_path=final_doors_dataset_path)
    train_set, validation_set = dataset_creator.create_datasets(split, max_samples, doors_config)
    labels = dataset_creator.get_labels()

    return train_set, validation_set, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
