from typing import List
import numpy as np
import pandas as pd
from typing import Tuple
import torch
from PIL import Image
import cv2

from generic_dataset.dataset_manager import DatasetManager
from doors_detection_long_term.doors_detector.dataset.dataset_igibson.door_sample import DoorSample as iGibsonDoorSample
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample import DoorSample as GibsonDoorSample
from doors_detection_long_term.doors_detector.dataset.torch_dataset import TorchDataset, SET

class DatasetDoorsIGibson(TorchDataset):
    def __init__(self, dataset_path: str, dataframe: pd.DataFrame, set_type: SET, std_size: int, max_size: int, scales: List[int]):
        super(DatasetDoorsIGibson, self).__init__(dataset_path, dataframe, set_type, std_size, max_size, scales)

        self._doors_dataset = DatasetManager(dataset_path=dataset_path, sample_class=iGibsonDoorSample)

    def load_sample(self, idx) -> Tuple[iGibsonDoorSample, str, int]:
        frame_row = self._dataframe.iloc[idx] # gets items by count (not index)
        folder_name, absolute_count = frame_row.folder_name, frame_row.folder_absolute_count

        loaded_door_sample: iGibsonDoorSample = self._doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        return loaded_door_sample, folder_name, absolute_count

    def __getitem__(self, idx):
        if idx >= len(self._dataframe.index):
            raise StopIteration

        door_sample, folder_name, absolute_count = self.load_sample(idx)
        sample_rgb_image = door_sample.get_rgb_image()
        sample_bgr_image = cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR)
        sample_bounding_boxes = door_sample.get_bounding_boxes()
        target = {}

        ## Filtering data
        filtered_bbox = self.filter_by_area(door_sample, 20)
        sample_bounding_boxes = filtered_bbox
        door_sample.set_bounding_boxes(filtered_bbox)
        filtered_bbox = self.filter_by_width(door_sample, 5)
        sample_bounding_boxes = filtered_bbox
        door_sample.set_bounding_boxes(filtered_bbox)
        invalid_data = not self.filter_by_rateo(door_sample, 0.15) or len(sample_bounding_boxes) == 0

        if invalid_data:
            self._dataframe.drop(self._dataframe.iloc[idx].name, axis=0, inplace=True)
            return self.__getitem__(idx)

        img_height, img_width, _ = sample_rgb_image.shape
        target["size"] = torch.tensor([int(img_height), int(img_width)], dtype=torch.int)
        target["boxes"] = torch.tensor(np.array([(x, y, x+w, y+h) for x, y, w, h, *_ in sample_bounding_boxes]), dtype=torch.float)
        target["labels"] = torch.tensor([box[4] for box in sample_bounding_boxes], dtype=torch.long)
        target["folder_name"] = folder_name
        target["absolute_count"] = absolute_count

        sample_image, target = self._transform(Image.fromarray((sample_bgr_image*255).astype(np.uint8)), target)

        return sample_image, target, door_sample
    
    def filter_by_area(self, sample, area_threshold):
        """
        Filters out bounding boxes with area smaller than area_threshold
        """
        bounding_boxes = sample.get_bounding_boxes()
        
        valid_bbs = list()
        for box in bounding_boxes:
            if box[2] * box[3] >= area_threshold:
                valid_bbs.append(box)

        return valid_bbs
    
    def filter_by_width(self, sample, width_threshold):
        """
        Filters out bounding boxes with smaller width than width_threshold and wich are in the margin
        of the image, usually representing doors cut off the frame
        """
        bounding_boxes = sample.get_bounding_boxes()
        image_width = sample.get_rgb_image().shape[1]

        valid_bbs = list()
        for box in bounding_boxes:
            is_on_margin = box[0] == 0 or box[0]+box[2] == image_width-1
            is_smaller = box[2] < width_threshold
            if not (is_on_margin and is_smaller):
                valid_bbs.append(box)

        return valid_bbs
    
    def filter_by_rateo(self, sample, rateo_threshold):
        """
        Filters out image which contains bounding boxes associated only to door frame and not to main
        part of the door, usually characterized by low width/height rateo
        """
        bounding_boxes = sample.get_bounding_boxes()

        for box in bounding_boxes:
            if box[2]/box[3] < rateo_threshold:
                return False
            
        return True
    
class DatasetDoorsIGibsonGibson(DatasetDoorsIGibson):
    def __init__(self, igibson_dataset_path:str, gibson_dataset_path:str, dataframe:pd.DataFrame, set_type: SET, std_size: int, max_size: int, scales: List[int]):
        super(DatasetDoorsIGibson, self).__init__(igibson_dataset_path, dataframe, set_type, std_size, max_size, scales)

        self._igibson_doors_dataset = DatasetManager(dataset_path=igibson_dataset_path, sample_class=iGibsonDoorSample)
        self._gibson_doors_dataset = DatasetManager(dataset_path=gibson_dataset_path, sample_class=GibsonDoorSample)

    def load_sample(self, idx) -> Tuple[iGibsonDoorSample, str, int]:
        frame_row = self._dataframe.iloc[idx] # gets items by count (not index)
        folder_name, absolute_count = frame_row.folder_name, frame_row.folder_absolute_count

        if frame_row.dataset == 'igibson':
            loaded_door_sample: iGibsonDoorSample = self._igibson_doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)
        else: # gibson_dataset
            gibson_sample: GibsonDoorSample = self._gibson_doors_dataset.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)
            loaded_door_sample = iGibsonDoorSample()

            ## adapting sample
            loaded_door_sample.set_rgb_image(gibson_sample.get_bgr_image())
            gibson_bboxes = gibson_sample.get_bounding_boxes()
            adapted_bboxes = list()
            for bbox in gibson_bboxes:
                adapted_box = (bbox[1], bbox[2], bbox[3], bbox[4], bbox[0], None, None)
                adapted_bboxes.append(adapted_box)
            loaded_door_sample.set_bounding_boxes(adapted_bboxes)

        return loaded_door_sample, folder_name, absolute_count

    def __getitem__(self, idx):
        if idx >= len(self._dataframe.index):
            raise StopIteration

        door_sample, folder_name, absolute_count = self.load_sample(idx)
        sample_rgb_image = door_sample.get_rgb_image()
        sample_bgr_image = cv2.cvtColor(sample_rgb_image, cv2.COLOR_RGB2BGR)
        sample_bounding_boxes = door_sample.get_bounding_boxes()
        target = {}

        if folder_name == 'igibson_dataset':
            ## Filtering data
            filtered_bbox = self.filter_by_area(door_sample, 20)
            sample_bounding_boxes = filtered_bbox
            door_sample.set_bounding_boxes(filtered_bbox)
            filtered_bbox = self.filter_by_width(door_sample, 5)
            sample_bounding_boxes = filtered_bbox
            door_sample.set_bounding_boxes(filtered_bbox)
            invalid_data = not self.filter_by_rateo(door_sample, 0.15) or len(sample_bounding_boxes) == 0

            if invalid_data:
                self._dataframe.drop(self._dataframe.iloc[idx].name, axis=0, inplace=True)
                return self.__getitem__(idx)
            
        img_height, img_width, _ = sample_rgb_image.shape
        target["size"] = torch.tensor([int(img_height), int(img_width)], dtype=torch.int)
        target["boxes"] = torch.tensor(np.array([(x, y, x+w, y+h) for x, y, w, h, *_ in sample_bounding_boxes]), dtype=torch.float)
        target["labels"] = torch.tensor([box[4] for box in sample_bounding_boxes], dtype=torch.long)
        target["folder_name"] = folder_name
        target["absolute_count"] = absolute_count

        sample_image, target = self._transform(Image.fromarray((sample_bgr_image*255).astype(np.uint8)), target)

        return sample_image, target, door_sample
    