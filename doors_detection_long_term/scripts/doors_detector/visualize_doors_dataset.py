from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.dataset_manager import DatasetManager

from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from dataset_configurator import deep_doors_2_labelled_dataset_path, final_doors_dataset_path
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample
params = {
    'seed': 0
}

#dataset_path = deep_doors_2_labelled_dataset_path
dataset_path = final_doors_dataset_path

if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    folder_manager = DatasetFolderManager(dataset_path=dataset_path, sample_class=DoorSample, folder_name='house1')

    for i in range(50):
        sample = folder_manager.load_sample_using_relative_count(label=1, relative_count=i, use_thread=False)
        sample.visualize()

