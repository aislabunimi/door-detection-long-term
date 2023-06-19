from abc import abstractmethod
from typing import Type, List, Tuple

import cv2
from PIL import Image
import numpy as np
import torch
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat

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


class TorchDatasetBBoxes(Dataset):
    def __init__(self, num_boxes: int):

        self._num_boxes = num_boxes

    def __len__(self):
        return len(self._images)

    def add_example_from_yolo(self, images, targets, preds, imgs_size):
        for image in images:
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            self._images.append(image)


        img_count_temp = self._img_count
        for target in targets:
            gt_boxes = []
            for label, [x, y, w, h] in zip(target['labels'].tolist(), target['boxes'].tolist()):
                gt_boxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x - w / 2, y - h / 2, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                ))
            self._gt_boxes.append(gt_boxes)
            self._img_count += 1


        assert len(preds[0]) <= self._num_boxes
        for boxes in preds:
            detected_boxes = []
            for x1, y1, x2, y2, score, label in boxes:
                label, score, = int(label.item()), score.item(),
                x1, y1, x2, y2 = x1.item() / imgs_size[0], y1.item() / imgs_size[1], x2.item() / imgs_size[0], y2.item() / imgs_size[1]
                if label >= 0:
                    detected_boxes.append(BoundingBox(
                        image_name=str(img_count_temp),
                        class_id=str(label),
                        coordinates=(x1, y1, x2 - x1, y2 - y1),
                        bb_type=BBType.DETECTED,
                        format=BBFormat.XYWH,
                        confidence=score
                    ))
            self._detected_bboxes.append(detected_boxes)
            img_count_temp += 1


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

        #bboxes = boxes / [(w, h, w, h) for _ in range(len(boxes))]

        target['boxes'] = torch.tensor(boxes, dtype=torch.float)
        target['labels'] = torch.tensor([label for label, *box in door_sample.get_bounding_boxes()], dtype=torch.long)
        target['folder_name'] = folder_name
        target['absolute_count'] = absolute_count

        # The BGR image is convert in RGB
        img, target = self._transform(Image.fromarray(door_sample.get_bgr_image()[..., [2, 1, 0]]), target)

        return img, target, door_sample