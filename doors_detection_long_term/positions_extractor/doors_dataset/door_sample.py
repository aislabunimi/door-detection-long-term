import os
from typing import List, NoReturn, Tuple

import cv2
import numpy as np
import pandas as pd
from generic_dataset.data_pipeline import DataPipeline
from generic_dataset.generic_sample import synchronize_on_fields
from generic_dataset.sample_generator import SampleGenerator
from generic_dataset.utilities.color import Color
from generic_dataset.utilities.save_load_methods import save_compressed_dictionary, load_compressed_dictionary, load_cv2_image_grayscale, save_cv2_image_bgr, load_cv2_image_bgr, save_compressed_numpy_array, load_compressed_numpy_array


pipeline_fix_gbr_image = DataPipeline().add_operation(lambda d, e: (d[..., [2, 1, 0]], e))


def round(data, engine):
    data[data > 10.0] = 10.0
    return data, engine
pipeline_generate_depth_image = DataPipeline().add_operation(operation=round) \
    .add_operation(round).add_operation(lambda data, engine: ((data * (255.0 / 10.0)).astype(engine.uint8), engine))


pipeline_fix_semantic_image = DataPipeline().add_operation(lambda d, e: (d.astype(e.uint8), e)).add_operation(lambda d, e: (d[..., [2, 1, 0]], e))


def get_matterport_positive_colors():
    dataframe = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'matterport_semantic_labels.tsv'), sep='\t')
    doors_rows = dataframe.loc[dataframe["mpcat40index"] == 4]
    doors_indexes = doors_rows["index"].to_list()
    return [Color(red=(index >> 16) % 256, green=(index >> 8) % 256, blue=index % 256).BGR() for index in doors_indexes]


positive_colors = get_matterport_positive_colors() + [Color(red=113, green=143, blue=65).BGR()]


@synchronize_on_fields(field_names={'pretty_semantic_image'}, check_pipeline=True)
def create_pretty_semantic_image(self, color: Color) -> 'DoorSample':
    """
    Creates the pretty semantic image starting from semantic image.
    :param color: the color used to fill positive pixels
    :return:
    """
    pretty_image = self.get_pretty_semantic_image()

    positive_colorss = self.get_positive_colors()
    for index in np.ndindex (pretty_image.shape[:2]):
        if pretty_image[index].tolist() in positive_colorss:
            pretty_image[index] = color.BGR()
        else:
            pretty_image[index] = 0

    self.set_pretty_semantic_image(pretty_image.astype(np.uint8))

    return self


@synchronize_on_fields(field_names={'semantic_image', 'label'}, check_pipeline=True)
def is_positive(self, threshold: float):
    """
    Changes the sample label according to the number of positive pixels.
    :param threshold:
    :return:
    """

    semantic_image = self.get_semantic_image()
    positive_colorss = self.get_positive_colors()
    positive_pixels = 0

    for index in np.ndindex(semantic_image.shape[:2]):
        if semantic_image[index].tolist() in positive_colorss:
            positive_pixels += 1

    if positive_pixels > semantic_image.shape[0] * semantic_image.shape[1] * threshold / 100:
        self.set_label(1)
    else:
        self.set_label(0)


@synchronize_on_fields(field_names={'bgr_image', 'depth_image', 'semantic_image', 'pretty_semantic_image', 'robot_pose', 'label'}, check_pipeline=True)
def visualize(self) -> NoReturn:
    """
    This method visualizes the sample, showing all its fields. Remember to calculates all fields before calling this method.
    :return:
    """
    print(f'Label: {self.get_label()}')
    print(f'Robot pose: {self.get_robot_pose()}')
    bgr_image = self.get_bgr_image()
    depth_image = self.get_depth_image()
    semantic_image = self.get_semantic_image()
    pretty_semantic_image = self.get_pretty_semantic_image()

    row_1 = np.concatenate((bgr_image, cv2.cvtColor(depth_image, cv2.COLOR_GRAY2BGR)), axis=1)
    row_2 = np.concatenate((semantic_image, pretty_semantic_image), axis=1)
    image = np.concatenate((row_1, row_2), axis=0)

    cv2.imshow('Sample', image)
    cv2.waitKey()


@synchronize_on_fields(field_names={'pretty_semantic_image'}, check_pipeline=True)
def get_bboxes(self, threshold: float = 0.0) -> List[Tuple[int, int, int, int]]:
    """
    Returns a list containing the bounding boxes calculated examining semantic image.
    :param threshold: parameter used to filter the bounding boxes that are too small
    :return:
    """
    rects = []
    pretty_image = self.get_pretty_semantic_image()
    _, threshed = cv2.threshold(cv2.cvtColor(pretty_image, cv2.COLOR_BGR2GRAY), thresh=30, maxval=255, type=cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshed, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        rect = cv2.boundingRect(contour)
        if rect[2] * rect[3] >= pretty_image.shape[0] * pretty_image.shape[1] * threshold:
            rects.append(rect)

    return rects
# The field bounding_boxes is a numpy array of tuple [(label, (x1, y1, width, height)],
# where label is the bounding box label and (x1, y1) are the coordinates of the top left point and width height the bbox dimension

DOOR_LABELS = {0: 'Closed door', 1: 'Semi opened door', 2: 'Opened door'}

DoorSample = SampleGenerator(name='DoorSample', label_set={0, 1}) \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray, save_function=save_cv2_image_bgr, load_function=load_cv2_image_bgr) \
    .add_custom_pipeline(method_name='pipeline_fix_bgr_image', elaborated_field='bgr_image', final_field='bgr_image', pipeline=pipeline_fix_gbr_image) \
    .add_dataset_field(field_name='depth_data', field_type=np.ndarray, save_function=save_compressed_numpy_array, load_function=load_compressed_numpy_array) \
    .add_field(field_name='depth_image', field_type=np.ndarray) \
    .add_custom_pipeline(method_name='pipeline_depth_data_to_image', elaborated_field='depth_data', final_field='depth_image', pipeline=pipeline_generate_depth_image) \
    .add_dataset_field(field_name='semantic_image', field_type=np.ndarray, save_function=save_cv2_image_bgr, load_function=load_cv2_image_bgr) \
    .add_custom_pipeline(method_name='pipeline_fix_semantic_image', elaborated_field='semantic_image', final_field='semantic_image', pipeline=pipeline_fix_semantic_image) \
    .add_field(field_name='positive_colors', field_type=List[List[int]], default_value=positive_colors) \
    .add_field(field_name='pretty_semantic_image', field_type=np.ndarray) \
    .add_custom_method(method_name='create_pretty_semantic_image', function=create_pretty_semantic_image) \
    .add_dataset_field(field_name='robot_pose', field_type=dict, save_function=save_compressed_dictionary, load_function=load_compressed_dictionary) \
    .add_dataset_field(field_name='bounding_boxes', field_type=np.ndarray, default_value=np.array([]), load_function=load_compressed_numpy_array, save_function=save_compressed_numpy_array) \
    .add_custom_method(method_name='calculate_positiveness', function=is_positive) \
    .add_custom_method(method_name='visualize', function=visualize) \
    .add_custom_method(method_name='get_bboxes_from_semantic_image', function=get_bboxes) \
    .generate_sample_class()