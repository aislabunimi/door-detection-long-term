import cv2
import numpy as np
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.utilities.color import Color
from generic_dataset.dataset_manager import DatasetManager

from doors_detection_long_term.positions_extractor.doors_dataset.door_sample import DoorSample


dataset_path ='/home/michele/myfiles/doors_dataset'

# Create the DatasetFolderManager instance and read sample
folder_manager = DatasetFolderManager(dataset_path=dataset_path, folder_name='house1', sample_class=DoorSample)

# Load a sample (positive, label = 1)
sample: DoorSample = folder_manager.load_sample_using_relative_count(label=1, relative_count=0, use_thread=False)
sample.set_pretty_semantic_image(sample.get_semantic_image().copy())
sample.pipeline_depth_data_to_image().run(use_gpu=False).get_data()
sample.create_pretty_semantic_image(color=Color(red=0, green=255, blue=0))

display_image_0 = np.concatenate((sample.get_bgr_image(), cv2.cvtColor(sample.get_depth_image(), cv2.COLOR_GRAY2BGR)), axis=1)
display_image_1 = np.concatenate((sample.get_semantic_image(), sample.get_pretty_semantic_image()), axis=1)

cv2.imshow('sample', np.concatenate((display_image_0, display_image_1), axis=0))
cv2.waitKey()

# Create DatasetManager instance and display dataset information
dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)

# Save the folders' metadata to disk
dataset.save_metadata()

print('The total amount of examples are')
for label, count in dataset.get_sample_count().items():
    print(' - {0} -> {1} samples'.format(label, count))