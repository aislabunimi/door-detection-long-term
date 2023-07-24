from typing import List
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from PIL import Image
import cv2

from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.door_sample import DoorSample
from doors_detection_long_term.doors_detector.dataset.torch_dataset import TorchDataset, SET

class DatasetDoorsIGibson(TorchDataset):
    def __init__(self, dataset_path: str, dataframe: pd.DataFrame, set_type: SET, std_size: int, max_size: int, scales: List[int]):
        super(DatasetDoorsIGibson, self).__init__(dataset_path, dataframe, set_type, std_size, max_size, scales)

        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=DoorSample)

    def load_sample(self, idx) -> Tuple[DoorSample, str, int]:
        frame_row = self._dataframe.iloc[idx]
        folder_name, absolute_count = frame_row.folder_name, frame_row.folder_absolute_count

        loaded_door_sample: DoorSample = self._doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        return loaded_door_sample, folder_name, absolute_count

    def __getitem__(self, idx):
        door_sample, folder_name, absolute_count = self.load_sample(idx)
        sample_rgb_image = door_sample.get_rgb_image()
        sample_bgr_image = cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR)
        sample_bounding_boxes = door_sample.get_bounding_boxes()
        target = {}

        img_height, img_width, _ = sample_rgb_image.shape
        target["size"] = torch.tensor([int(img_height), int(img_width)], dtype=torch.int)
        target["boxes"] = torch.tensor(np.array([(x, y, x+w, y+h) for x, y, w, h, *_ in sample_bounding_boxes]), dtype=torch.float)
        target["labels"] = torch.tensor([box[4] for box in sample_bounding_boxes], dtype=torch.long)
        target["folder_name"] = folder_name
        target["absolute_count"] = absolute_count

        sample_image, target = self._transform(Image.fromarray((sample_bgr_image*255).astype(np.uint8)), target)

        return sample_image, target, door_sample