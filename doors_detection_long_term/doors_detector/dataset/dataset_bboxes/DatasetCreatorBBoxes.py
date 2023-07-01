from enum import Enum

import cv2
import numpy as np
import torch
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat, CoordinatesType
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TorchDatasetBBoxes, TRAIN_SET, TEST_SET


class Type(Enum):
    TRAINING = 1
    TEST = 2

class DatasetsCreatorBBoxes:
    def __init__(self,  num_bboxes):
        self._colors = {0: (0, 0, 255), 1: (0, 255, 0)}
        self._num_bboxes = num_bboxes

        self._img_count = 0
        self._training_bboxes = {
            'images': [],
            'gt_bboxes': [],
            'bboxes': [],
            'filtered': []
        }
        self._test_bboxes = {
            'images': [],
            'gt_bboxes': [],
            'bboxes': [],
            'filtered': []
        }

    def visualize_bboxes(self, show_filtered: bool = False, bboxes_type: Type = Type.TRAINING):
        if bboxes_type == Type.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == Type.TRAINING:
            bboxes_dict = self._training_bboxes
        for i, (image, bboxes, target_bboxes) in enumerate(zip(bboxes_dict['images'], bboxes_dict['bboxes'], bboxes_dict['gt_bboxes'])):
            img_size = image.shape
            show_image = image.copy()
            target_image = image.copy()
            filtered_image = image.copy()
            for bbox in target_bboxes:
                x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
                print(x1, y1, x2, y2)
                target_image = cv2.rectangle(target_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(bbox.get_class_id())], 2)
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
                show_image = cv2.rectangle(show_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(bbox.get_class_id())], 2)
            if show_filtered:
                for y, bbox in enumerate(bboxes):
                    if bboxes_dict['filtered'][i][y] == 1:
                        x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
                        filtered_image = cv2.rectangle(filtered_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(bbox.get_class_id())], 2)
            show_image = cv2.hconcat([target_image, show_image, filtered_image])
            cv2.imshow('show', show_image)
            cv2.waitKey()

    def filter_bboxes(self, iou_threshold: float = 0.75, filter_multiple_detection: bool = False, consider_label: bool = False):
        check_label= lambda gt_label, detected_label: True if not consider_label else gt_label == detected_label

        def _filtered_bboxes(bboxes_dict):
            for image_count, (targets, detected) in enumerate(zip(bboxes_dict['gt_bboxes'], bboxes_dict['bboxes'])):
                filtered = [0 for _ in range(len(detected))]
                # Like the AP, the multiple detection of the same object are removed
                # The matched bbox is the one with the high IoU area
                if filter_multiple_detection:
                    for target_bbox in targets:
                        max_iou = -1
                        max_index = -1
                        for i, detected_bbox in enumerate(detected):
                            iou_area = BoundingBox.iou(detected_bbox, target_bbox)
                            if iou_area >= iou_threshold and \
                                iou_area >= max_iou and \
                                check_label(target_bbox.get_class_id(), detected_bbox.get_class_id()):
                                max_iou = iou_area
                                max_index = i

                        if max_index > -1:
                            filtered[max_index] = 1
                else:
                    for i, detected_bbox in enumerate(detected):
                        for target_bbox in targets:
                            iou_area = BoundingBox.iou(detected_bbox, target_bbox)
                            if iou_area >= iou_threshold and check_label(target_bbox.get_class_id(), detected_bbox.get_class_id()):
                                filtered[i] = 1
                bboxes_dict['filtered'].append(filtered)

        _filtered_bboxes(self._training_bboxes)
        _filtered_bboxes(self._test_bboxes)


    def _filter_images_no_boxes(self, bboxes_dict):
        """
        Remove those images with no valid bounding boxes.
        To the valid images, adds dummy bboxes to reach the bbox number
        :param bboxes_dict:
        :return:
        """
        bboxes_dict_converted = {
            'images': [],
            'bboxes': [],
            'filtered': [],
            'gt_bboxes': []
        }

        for count, (image, detected_boxes, filtered, gt_bboxes) in enumerate(zip(bboxes_dict['images'], bboxes_dict['bboxes'], bboxes_dict['filtered'], bboxes_dict['gt_bboxes'])):

            # If the example do not have valid bboxes, it is discarded
            if 1 not in filtered:
                continue

            # Add more fake boxes if the number of total bboxes is less than self._num_boxes
            for _ in range(len(filtered), self._num_bboxes):
                filtered.append(0)
                detected_boxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id='0',
                    confidence=1.0,
                    coordinates=(0, 0, 0, 0),
                    bb_type=BBType.DETECTED,
                    format=BBFormat.XYWH,
                    img_size=image.shape[1:][::-1]
                ))

            assert len(filtered) == len(detected_boxes) and len(filtered) == self._num_bboxes

            bboxes_dict_converted['images'].append(image)
            bboxes_dict_converted['filtered'].append(filtered)
            bboxes_dict_converted['bboxes'].append(detected_boxes)
            bboxes_dict_converted['gt_bboxes'].append(gt_bboxes)
        return bboxes_dict_converted

    def _shuffle_bboxes_in_images(self, bboxes_dict, num_shuffles):
        bboxes_dict_converted = {
            'images': [],
            'bboxes': [],
            'filtered': [],
            'gt_bboxes': []
        }
        for image, detected_boxes, filtered, gt_bboxes in zip(bboxes_dict['images'], bboxes_dict['bboxes'], bboxes_dict['filtered'],bboxes_dict['gt_bboxes']):
            bboxes_dict_converted['images'].append(image)
            bboxes_dict_converted['bboxes'].append(detected_boxes)
            bboxes_dict_converted['filtered'].append(filtered)
            bboxes_dict_converted['gt_bboxes'].append(gt_bboxes)
            for i in range(num_shuffles):
                bboxes_dict_converted['images'].append(image)
                new_detected_boxes, new_filtered = shuffle(detected_boxes, filtered, random_state=i)
                bboxes_dict_converted['bboxes'].append(new_detected_boxes)
                bboxes_dict_converted['filtered'].append(new_filtered)
                bboxes_dict_converted['gt_bboxes'].append(gt_bboxes)
        return bboxes_dict_converted


    def create_datasets(self, random_state: int = 42):

        bboxes_dict_train = self._filter_images_no_boxes(self._training_bboxes)
        bboxes_dict_train = self._shuffle_bboxes_in_images(bboxes_dict_train, num_shuffles=5)
        bboxes_dict_test = self._filter_images_no_boxes(self._test_bboxes)

        images, bboxes, filtered, gt_bboxes = shuffle(bboxes_dict_train['images'],
                                           bboxes_dict_train['bboxes'],
                                           bboxes_dict_train['filtered'],
                                           bboxes_dict_train['gt_bboxes'],
                                           random_state=random_state)

        bboxes_dict_train = {
            'images': images,
            'bboxes': bboxes,
            'filtered': filtered,
            'gt_bboxes': gt_bboxes
        }

        images, bboxes, filtered, gt_bboxes = shuffle(bboxes_dict_test['images'],
                                           bboxes_dict_test['bboxes'],
                                           bboxes_dict_test['filtered'],
                                           bboxes_dict_test['gt_bboxes'],
                                           random_state=random_state)

        bboxes_dict_test = {
            'images': images,
            'bboxes': bboxes,
            'filtered': filtered,
            'gt_bboxes': gt_bboxes
        }

        return (TorchDatasetBBoxes(bboxes_dict=bboxes_dict_train, num_boxes=self._num_bboxes, set_type=TRAIN_SET),
                TorchDatasetBBoxes(bboxes_dict=bboxes_dict_test, num_boxes=self._num_bboxes, set_type=TEST_SET))


    def add_yolo_bboxes(self, images, targets, preds, bboxes_type: Type):
        if bboxes_type == Type.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == Type.TRAINING:
            bboxes_dict = self._training_bboxes

        img_size = images.size()[2:][::-1]
        for image in images:
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            bboxes_dict['images'].append(image)


        img_count_temp = self._img_count
        for target in targets:
            gt_boxes = []
            for label, [x, y, w, h] in zip(target['labels'].tolist(), target['boxes'].tolist()):
                gt_boxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x, y, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                    img_size=img_size,
                    type_coordinates=CoordinatesType.RELATIVE,
                ))
            bboxes_dict['gt_bboxes'].append(gt_boxes)
            self._img_count += 1


        assert len(preds[0]) <= self._num_bboxes

        for boxes in preds:
            detected_boxes = []
            for x1, y1, x2, y2, score, label in boxes.tolist():
                label = int(label)
                if label >= 0:
                    detected_boxes.append(BoundingBox(
                        image_name=str(img_count_temp),
                        class_id=str(label),
                        type_coordinates=CoordinatesType.ABSOLUTE,
                        coordinates=(x1, y1, x2 - x1, y2 - y1),
                        bb_type=BBType.DETECTED,
                        format=BBFormat.XYWH,
                        confidence=score,
                        img_size=img_size
                    ))
            bboxes_dict['bboxes'].append(detected_boxes)
            img_count_temp += 1