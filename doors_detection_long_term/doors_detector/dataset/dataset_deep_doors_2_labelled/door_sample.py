from typing import NoReturn

import cv2
import numpy as np
from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.generic_sample import synchronize_on_fields
from generic_dataset.sample_generator import SampleGenerator
from generic_dataset.utilities.save_load_methods import save_cv2_image_bgr, load_cv2_image_bgr, \
    save_compressed_numpy_array, load_compressed_numpy_array, load_cv2_image_grayscale

COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]

pipeline_fix_gbr_image = DataPipeline().add_operation(lambda d, e: (d[..., [2, 1, 0]], e))

@synchronize_on_fields(field_names={'bgr_image', 'depth_image', 'bounding_boxes'}, check_pipeline=True)
def visualize(self) -> NoReturn:
    """
    This method visualizes the sample, showing all its fields.
    :return:
    """
    bgr_image = self.get_bgr_image()
    depth_image = self.get_depth_image()
    img_bounding_boxes = bgr_image.copy()

    for label, *box in self.get_bounding_boxes():
        cv2.rectangle(img_bounding_boxes, box, color=COLORS[label], thickness=1)

    row_1 = np.concatenate((bgr_image, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)), axis=1)
    row_1 = np.concatenate((row_1, img_bounding_boxes), axis=1)

    cv2.imshow('Sample', row_1)
    cv2.waitKey()


# The bounding_boxes field is a numpy array of tuple [(label, x1, y1, width, height)],
# where label is the bounding box label and (x1, y1) are the coordinates of the top left point and width height the bbox dimension

DOOR_LABELS = {0: 'Closed door', 1: 'Semi-open door', 2: 'Open door'}

DoorSample = SampleGenerator(name='DoorSample', label_set={0, 1}) \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray, save_function=save_cv2_image_bgr, load_function=load_cv2_image_bgr) \
    .add_dataset_field(field_name='depth_image', field_type=np.ndarray, save_function=save_cv2_image_bgr, load_function=load_cv2_image_grayscale) \
    .add_dataset_field(field_name='bounding_boxes', field_type=np.ndarray, default_value=np.array([]), load_function=load_compressed_numpy_array, save_function=save_compressed_numpy_array) \
    .add_custom_method(method_name='visualize', function=visualize) \
    .generate_sample_class()