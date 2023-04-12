import numpy as np
import pandas as pd
import torch
from generic_dataset.dataset_manager import DatasetManager
from generic_dataset.utilities.color import Color
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample_real_data import DoorSample as DoorSampleRealData

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample as DoorSampleFinalDoorsDataset
from doors_detection_long_term.doors_detector.dataset.dataset_deep_doors_2_labelled.door_sample import DoorSample as DoorSampleDeepDoors2
from PIL import Image
from typing import Type, List, Tuple
from doors_detection_long_term.doors_detector.dataset.torch_dataset import TorchDataset, SET, TRAIN_SET, TEST_SET


class DatasetDoorsFinal(TorchDataset):
    def __init__(self, dataset_path: str,
                 dataframe: pd.DataFrame,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):

        super(DatasetDoorsFinal, self).__init__(
            dataset_path=dataset_path,
            dataframe=dataframe,
            set_type=set_type,
            std_size=std_size,
            max_size=max_size,
            scales=scales)

        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSampleFinalDoorsDataset)

    def load_sample(self, idx) -> Tuple[DoorSampleFinalDoorsDataset, str, int]:
        row = self._dataframe.iloc[idx]
        folder_name, absolute_count = row.folder_name, row.folder_absolute_count

        door_sample: DoorSampleFinalDoorsDataset = self._doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        return door_sample, folder_name, absolute_count


class DatasetDoorsFinalAndDeepDoors2(TorchDataset):
    def __init__(self, dataset_path_gibson: str,
                 dataset_path_deep_doors_2_relabelled: str,
                 dataframe: pd.DataFrame,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):

        super(DatasetDoorsFinalAndDeepDoors2, self).__init__(
            dataset_path=dataset_path_gibson,
            dataframe=dataframe,
            set_type=set_type,
            std_size=std_size,
            max_size=max_size,
            scales=scales)

        if dataset_path_gibson != '':
            self._doors_dataset_gibson = DatasetManager(dataset_path=dataset_path_gibson, sample_class=DoorSampleFinalDoorsDataset)
        if dataset_path_deep_doors_2_relabelled != '':
            self._doors_dataset_deep_doors_2 = DatasetManager(dataset_path=dataset_path_deep_doors_2_relabelled, sample_class=DoorSampleDeepDoors2)

    def load_sample(self, idx) -> Tuple[DoorSampleFinalDoorsDataset, str, int]:
        row = self._dataframe.iloc[idx]
        folder_name, absolute_count = row.folder_name, row.folder_absolute_count

        if row.folder_name == 'deep_doors_2':
            door_sample = DoorSampleFinalDoorsDataset()

            door_sample_deep_doors_2 = self._doors_dataset_deep_doors_2.load_sample(folder_name=folder_name, absolute_count=absolute_count)
            bboxes = door_sample_deep_doors_2.get_bounding_boxes()

            bboxes_fixed = []
            for [label, x1, x2, width, height] in bboxes:
                bboxes_fixed.append([max(0, label - 1), x1, x2, width, height])

            door_sample.set_bounding_boxes(np.array(bboxes_fixed))
            door_sample.set_bgr_image(door_sample_deep_doors_2.get_bgr_image())
        else:
            door_sample: DoorSampleFinalDoorsDataset = self._doors_dataset_gibson.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        return door_sample, folder_name, absolute_count

class DatasetDoorsFinalRealData(TorchDataset):
    def __init__(self, dataset_path: str,
                 dataframe: pd.DataFrame,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):

        super(DatasetDoorsFinalRealData, self).__init__(
            dataset_path=dataset_path,
            dataframe=dataframe,
            set_type=set_type,
            std_size=std_size,
            max_size=max_size,
            scales=scales)

        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSampleRealData)

    def load_sample(self, idx) -> Tuple[DoorSampleFinalDoorsDataset, str, int]:
        row = self._dataframe.iloc[idx]
        folder_name, absolute_count = row.folder_name, row.folder_absolute_count

        door_sample: DoorSampleFinalDoorsDataset = self._doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        return door_sample, folder_name, absolute_count