import numpy as np
from typing import NoReturn
import cv2

from generic_dataset.sample_generator import SampleGenerator
import generic_dataset.utilities.save_load_methods as slm
from generic_dataset.generic_sample import synchronize_on_fields
from . import dataset_methods as dsm

COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}
DOOR_LABELS = {0: "Closed door", 1: "Open door"}

@synchronize_on_fields(field_names={"rgb_image", "depth_image", "semantic_image", "bounding_boxes"}, check_pipeline=True)
def show_sample(self) -> NoReturn:
    rgb_image = self.get_rgb_image()
    depth_image = self.get_depth_image()
    semantic_image = self.get_samantic_image()
    bounding_boxes = self.get_bounding_boxes()

    boxes_image = rgb_image.copy()
    for bounding_box in bounding_boxes:
        cv2.rectangle(boxes_image, bounding_box[:4], COLORS[bounding_box[4]], 1)

    row_1 = np.concatenate((rgb_image, boxes_image), axis=1)
    row_2 = np.concatenate((depth_image, semantic_image), axis=1)
    grid = np.concatenate((row_1, row_2), axis=0)

    cv2.imshow('Sample', grid)
    cv2.waitKey()

DoorSample = SampleGenerator(name="IGibsonSample", label_set={0, 1}) \
    .add_dataset_field(field_name="rgb_image", field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=dsm.load_compressed_numpy_array) \
    .add_dataset_field(field_name="depth_image", field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=dsm.load_compressed_numpy_array) \
    .add_dataset_field(field_name="semantic_image", field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=dsm.load_compressed_numpy_array) \
    .add_dataset_field(field_name="bounding_boxes", field_type=list, save_function=dsm.save_tuple_list, load_function=dsm.load_tuple_list) \
    .add_dataset_field(field_name="semantic_instance_image", field_type=np.ndarray, save_function=slm.save_compressed_numpy_array, load_function=dsm.load_compressed_numpy_array) \
    .add_custom_method(method_name='show_sample', function=show_sample) \
    .generate_sample_class()

## bounding_boxes field contains a list of tuples [(x, y, w, h, label, ...unused values), ...], where
## x and y are the coordinates of the top-left corner of the bounding box, w and h are its width and height
