from abc import abstractmethod
import random
from typing import Type, List, Tuple

import cv2
from PIL import Image
import numpy as np
import torch
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.dataset_manager import DatasetManager
from sklearn.utils import shuffle
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
    def __init__(self, dataset_manager: DatasetManager, folder_name: str,dataframe, set_type, max_bboxes: int, iou_threshold_matching: float, shuffle_boxes: bool = False):

        self._dataset_manager = dataset_manager
        self._folder_name = folder_name
        self._dataframe = dataframe
        self._shuffle = shuffle_boxes
        self._max_bboxes = max_bboxes
        self._iou_threshold_matching = iou_threshold_matching

        scales = [256 + i * 32 for i in range(7)]

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
        return len(self._dataframe.index)

    def __getitem__(self, idx):
        row = self._dataframe.iloc[idx]
        folder_name, absolute_count = row.folder_name, row.folder_absolute_count

        sample = self._dataset_manager.load_sample(folder_name=folder_name, absolute_count=absolute_count, use_thread=False)

        # Read bboxes
        gt_bboxes = []
        for x1, y1, x2, y2, label, conf in sample.get_gt_bounding_boxes():
            gt_bboxes.append(BoundingBox(
                image_name='0',
                class_id=str(int(float(label))),
                coordinates=(int(x1), int(y1), int(x2), int(y2)),
                bb_type=BBType.GROUND_TRUTH,
                format=BBFormat.XYX2Y2,
            ))
        print(gt_bboxes)
        bboxes = []
        for x1, y1, x2, y2, label, conf in sample.get_detected_bounding_boxes():
            bboxes.append(BoundingBox(
                image_name='0',
                class_id=str(int(float(label))),
                coordinates=(int(x1), int(y1), int(x2), int(y2)),
                bb_type=BBType.DETECTED,
                format=BBFormat.XYX2Y2,
                confidence=conf
            ))

        # Select the firsts n bboxes
        bboxes = sorted(bboxes, key=lambda x: x.get_confidence(), reverse=True)[:self._max_bboxes]

        # Matchong bboxes with gt
        matched_bboxes = []
        for detected_bbox in bboxes:
            matched_gt = None
            match_iou = -1
            for gt_bbox in gt_bboxes:
                iou = BoundingBox.iou(detected_bbox, gt_bbox)
                if iou >= match_iou and iou >= self._iou_threshold_matching:
                    matched_gt = gt_bbox
                    match_iou = iou
            matched_bboxes.append((detected_bbox, matched_gt))

        image = sample.get_bgr_image()

        target = {}
        (h, w, _) = image.shape
        target['size'] = torch.tensor([int(h), int(w)], dtype=torch.int)

        detected_boxes = []
        original_confidences = []
        original_labels = []

        target_boxes = []
        fixed_boxes = [] # the coordinates of the matched gt boxes
        confidences = []
        labels_encoded = []
        ious = []

        for gt_box in gt_bboxes:
            x, y, w, h = gt_box.get_absolute_bounding_box()
            original_label = int(gt_box.get_class_id())
            target_boxes.append([x, y, x + w, y + h, original_label])

        target_boxes = torch.tensor(target_boxes)

        for detected_box, gt_box in matched_bboxes:
            x, y, w, h = detected_box.get_absolute_bounding_box()
            original_label = int(detected_box.get_class_id())

            detected_boxes.append((x, y, x + w, y + h))
            original_confidences.append([detected_box.get_confidence()])
            original_labels.append([0 if i != original_label else 1 for i in range(2)])

            confidences.append(0.0 if gt_box is None else 1.0)
            label = 0 if gt_box is None else int(gt_box.get_class_id()) + 1
            labels_encoded.append([0 if i != label else 1 for i in range(3)]) # 3 is the number of label (background, closed door, open door)
            ious.append(0.0 if gt_box is None else BoundingBox.iou(detected_box, gt_box))
            gt_x, gt_y, gt_w, gt_h = gt_box.get_absolute_bounding_box() if gt_box is not None else (x, y, w, h)

            fixed_boxes.append((gt_x, gt_y, gt_x + gt_w, gt_y + gt_h))

        len_detected_bboxes = len(detected_boxes)
        len_fixed_bboxes = len(fixed_boxes)

        t = []
        if target_boxes.size()[0] > 0:
            t = target_boxes[:, :4].tolist()
        target['boxes'] = torch.tensor(detected_boxes + fixed_boxes + t, dtype=torch.float)

        #target['labels_encoded'] = torch.tensor(labels_encoded, dtype=torch.float)
        #target['confidences'] = torch.tensor(confidences, dtype=torch.float)
        #target['original_labels'] = torch.tensor(original_labels)
        #target['original_confidences'] = torch.tensor(original_confidences)
        #target['ious'] = torch.tensor(ious)

        # The BGR image is convert in RGB
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        img, target = self._transform(Image.fromarray(image[..., [2, 1, 0]]), target)

        detected_boxes = target['boxes'][:len_detected_bboxes].tolist()
        fixed_boxes = target['boxes'][len_detected_bboxes:len_detected_bboxes + len_fixed_bboxes].tolist()
        target_bboxes_modified = target['boxes'][len_detected_bboxes + len_fixed_bboxes:]

        if self._shuffle:
            random_state = random.randint(1, 1000)
            fixed_boxes, detected_boxes, labels_encoded, confidences, original_labels, original_confidences, ious = shuffle(
                np.array(fixed_boxes),
                np.array(detected_boxes),
                np.array(labels_encoded),
                np.array(confidences),
                np.array(original_labels),
                np.array(original_confidences),
                np.array(ious),
                random_state=random_state
            )

        target['labels_encoded'] = torch.tensor(labels_encoded, dtype=torch.float)
        target['confidences'] = torch.tensor(confidences, dtype=torch.float)
        target['original_labels'] = torch.tensor(original_labels, dtype=torch.float)
        target['original_confidences'] = torch.tensor(original_confidences, dtype=torch.float)
        target['ious'] = torch.tensor(ious, dtype=torch.float)
        target['fixed_boxes'] = torch.tensor(fixed_boxes, dtype=torch.float)
        target['detected_boxes'] = torch.tensor(detected_boxes, dtype=torch.float)
        if target_boxes.size()[0] > 0:
            target_boxes[:, :4] = target_bboxes_modified
        target['target_boxes'] = target_boxes
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