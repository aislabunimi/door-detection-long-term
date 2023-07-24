from abc import abstractmethod
from typing import Type, List, Tuple

import cv2
from PIL import Image
import numpy as np
import torch
#from src.bounding_box import BoundingBox
#from src.utils.enumerators import BBType, BBFormat

from doors_detection_long_term.positions_extractor.doors_dataset.door_sample import DoorSample

import doors_detection_long_term.doors_detector.utilities.transforms as T
import pandas as pd
from torch.utils.data import Dataset

SET = str
TRAIN_SET: SET = 'train_set'
TEST_SET: SET = 'test_set'

DATASET = str
DEEP_DOORS_2_LABELLED: DATASET = 'deep_doors_2_labelled'
FINAL_DOORS_DATASET: DATASET = 'final_doors_dataset'
BOUNDING_BOX_DATASET: DATASET = 'bounding_box_dataset'
IGIBSON_DATASET: DATASET = 'igibson_dataset'


class TorchDatasetBBoxes(Dataset):
    def __init__(self, bboxes_dict: dict, num_boxes: int, set_type: SET):

        self._bboxes_dict = bboxes_dict
        self._num_boxes = num_boxes
        scales = [256 + i * 32 for i in range(11)]

        if set_type == TEST_SET:
            self._transform = T.Compose([
                #T.RandomResize([std_size], max_size=max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:

            self._transform = T.Compose([
                T.RandomSelect(
                    T.Identity(),
                    T.Compose([
                        T.RandomHorizontalFlip(),
                        T.RandomResize(scales, max_size=800),
                    ])
                ),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self._bboxes_dict['images'])

    def __getitem__(self, idx):
        image, bboxes, filtered, gt_bboxes = self._bboxes_dict['images'][idx], self._bboxes_dict['bboxes'][idx], self._bboxes_dict['filtered'][idx], self._bboxes_dict['gt_bboxes'][idx]

        target = {}
        (h, w, _) = image.shape
        target['size'] = torch.tensor([int(h), int(w)], dtype=torch.int)

        # Normalize bboxes' size. The bboxes are initially defined as (x_top_left, y_top_left, width, height)
        # Bboxes representation changes, becoming a tuple (center_x, center_y, width, height).
        # All values must be normalized in [0, 1], relative to the image's size
        boxes = []
        confidences = []
        labels_encoded = []
        for box in bboxes:
            x, y, w, h = box.get_absolute_bounding_box()
            label = int(box.get_class_id())

            boxes.append((x, y, x + w, y + h))
            confidences.append(box.get_confidence())
            labels_encoded.append([0 if i != label else 1 for i in range(2)]) # 2 is the number of label

        target['boxes'] = torch.tensor(boxes, dtype=torch.float)
        target['labels_encoded'] = torch.tensor(labels_encoded, dtype=torch.float)
        target['confidences'] = torch.tensor(confidences, dtype=torch.float)
        target['filtered'] = torch.tensor(filtered)

        # The BGR image is convert in RGB
        image = (image * 255).astype(np.uint8)
        img, target = self._transform(Image.fromarray(image[..., [2, 1, 0]]), target)
        target['gt_bboxes'] = gt_bboxes
        return img, target


class TorchDataset(Dataset):
    def __init__(self, dataset_path: str, dataframe: pd.DataFrame, set_type: SET, std_size: int, max_size: int, scales: List[int]):
        self._dataset_path = dataset_path
        self._dataframe = dataframe
        self._set_type = set_type

        if set_type == TEST_SET:
            self._transform = T.Compose([
                #T.RandomResize([std_size], max_size=max_size),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:

            self._transform = T.Compose([
                T.RandomSelect(
                    T.Identity(),
                    T.Compose([
                        T.RandomHorizontalFlip(),
                        T.RandomResize(scales, max_size=max_size),
                    ])
                ),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def get_dataframe(self) -> pd.DataFrame:
        return self._dataframe

    def __len__(self):
        return len(self._dataframe.index)

    @abstractmethod
    def load_sample(self, idx) -> Tuple[DoorSample, str, int]:
        pass

    def __getitem__(self, idx):
        door_sample, folder_name, absolute_count = self.load_sample(idx)

        target = {}
        (h, w, _) = door_sample.get_bgr_image().shape
        target['size'] = torch.tensor([int(h), int(w)], dtype=torch.int)

        # Normalize bboxes' size. The bboxes are initially defined as (x_top_left, y_top_left, width, height)
        # Bboxes representation changes, becoming a tuple (center_x, center_y, width, height).
        # All values must be normalized in [0, 1], relative to the image's size
        boxes = door_sample.get_bounding_boxes()
        boxes = np.array([(x, y, x + w, y + h) for label, x, y, w, h in boxes])

        target['boxes'] = torch.tensor(boxes, dtype=torch.float)
        target['labels'] = torch.tensor([label for label, *box in door_sample.get_bounding_boxes()], dtype=torch.long)
        target['folder_name'] = folder_name
        target['absolute_count'] = absolute_count

        # The BGR image is convert in RGB
        img, target = self._transform(Image.fromarray(door_sample.get_bgr_image()[..., [2, 1, 0]]), target)

        return img, target, door_sample