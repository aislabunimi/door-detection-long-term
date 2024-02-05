import numpy as np
from generic_dataset.dataset_manager import DatasetManager

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
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final_bbox_filter_one_house import \
    DatasetsCreatorDoorsFinalBBoxFilterOneHouse
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final_epoch_analysis import \
    DatasetsCreatorDoorsFinalEpochAnalysis
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_no_door_task import \
    DatasetsCreatorDoorsNoDoorTask
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample_real_data import DoorSample

# The path in which the trained model are saved and loaded
# If the string is empty, they are saved in a folder in this repository (/models/train_params/)
trained_models_path = "/home/michele/myfiles/trained_models"

deep_doors_2_labelled_dataset_path = '/home/michele/myfiles/deep_doors_2_labelled'
final_doors_dataset_path = '/home/michele/myfiles/final_doors_dataset/'
real_final_doors_dataset_path = '/home/michele/myfiles/final_doors_dataset_real'
box_filtering_dataset_path = '/home/michele/myfiles/box_filtering_dataset'



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

def get_deep_doors_2_relabelled_dataset_for_gd(fixed_scale=False):
    dataset_creator = DatasetsCreatorDeepDoors2LabelledGD(dataset_path=deep_doors_2_labelled_dataset_path)

    train, validation = dataset_creator.creates_dataset(fixed_scale=fixed_scale)
    labels = dataset_creator.get_labels()

    return train, validation, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_gibson_and_deep_door_2_dataset(half: bool):
    dataset_creator = DatasetsCreatorGibsonAndDeepDoors2(dataset_path_gibson=final_doors_dataset_path, dataset_path_deep_doors_2=deep_doors_2_labelled_dataset_path)

    train, validation = dataset_creator.creates_dataset(half=half)
    labels = dataset_creator.get_labels()

    return train, validation, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_real_data(folder_name: str, train_size: float = 0.1, transform_train=True):
    dataset_creator = DatasetsCreatorRealData(dataset_path=real_final_doors_dataset_path)
    train, test = dataset_creator.create_datasets(folder_name=folder_name, train_size=train_size, transform_train=transform_train)
    labels = dataset_creator.get_labels()

    return train, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)


def get_final_doors_dataset_hybrid(folder_name_to_exclude: str, transform_train=False):
    dataset_creator = DatasetsCreatorAllEnvs(dataset_path=final_doors_dataset_path)
    train_total, validation_total = dataset_creator.create_datasets()

    dataset_creator = DatasetsCreatorDeepDoors2LabelledGD(dataset_path=deep_doors_2_labelled_dataset_path)
    train, validation = dataset_creator.creates_dataset()

    train_total += train
    validation_total += validation

    manager = DatasetManager(dataset_path=real_final_doors_dataset_path, sample_class=DoorSample)
    dataset_folders = [env for env in manager.get_folder_names() if 'evening' not in env and env != folder_name_to_exclude]
    for folder_name in dataset_folders:
        dataset_creator = DatasetsCreatorRealData(dataset_path=real_final_doors_dataset_path)
        train, validation = dataset_creator.create_datasets(folder_name=folder_name, train_size=0.75, transform_train=transform_train)

        train_total += train
        validation_total += validation
    labels = dataset_creator.get_labels()
    return train_total, validation_total, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_real_hybrid(folder_name_to_exclude: str, transform_train=False):

    train_total, test_total = None, None
    manager = DatasetManager(dataset_path=real_final_doors_dataset_path, sample_class=DoorSample)
    dataset_folders = [env for env in manager.get_folder_names() if 'evening' not in env and env != folder_name_to_exclude]
    for folder_name in dataset_folders:
        dataset_creator = DatasetsCreatorRealData(dataset_path=real_final_doors_dataset_path)
        train, test = dataset_creator.create_datasets(folder_name=folder_name, train_size=0.75, transform_train=transform_train)
        if train_total is None and test_total is None:
            train_total = train
            test_total = test
        else:
            train_total += train
            test_total += test
    labels = dataset_creator.get_labels()
    return train_total, test_total, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_bbox_filter(folder_name: str, train_size_student: float = .0):
    dataset_creator = DatasetsCreatorDoorsFinalBBoxFilter(dataset_path=final_doors_dataset_path)
    train_student, validation_student, unlabelled_bbox_filter, test = dataset_creator.create_datasets(folder_name=folder_name, train_size_student=train_size_student)
    labels = dataset_creator.get_labels()

    return train_student, validation_student, unlabelled_bbox_filter, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_bbox_filter_one_house(folder_name: str, use_negatives=False):
    dataset_creator = DatasetsCreatorDoorsFinalBBoxFilterOneHouse(dataset_path=final_doors_dataset_path)
    train, test = dataset_creator.create_datasets(folder_name=folder_name, use_negatives=use_negatives)
    labels = dataset_creator.get_labels()

    return train, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

def get_final_doors_dataset_real_data_bbox_filter(folder_name: str, train_size: float = 0.15, control_size: float = 0.15, fine_tune_size: float = 0.45, test_size: float = 0.25):
    dataset_creator = DatasetsCreatorRealDataBboxFilter(dataset_path=real_final_doors_dataset_path)
    train, control, fine_tune, test = dataset_creator.create_datasets(folder_name=folder_name, control_size=control_size, fine_tune_size=fine_tune_size, train_size=train_size, test_size=test_size )
    labels = dataset_creator.get_labels()

    return train, control, fine_tune, test, labels, np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

