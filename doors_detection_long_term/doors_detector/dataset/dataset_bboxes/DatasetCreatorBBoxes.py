from enum import Enum

import cv2
import numpy as np
import torch
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat, CoordinatesType


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
            'detected_bboxes': [],
            'filtered': []
        }
        self._test_bboxes = {
            'images': [],
            'gt_bboxes': [],
            'detected_bboxes': [],
            'filtered': []
        }

    def visualize_bboxes(self, show_filtered: bool = False, bboxes_type: Type = Type.TRAINING):
        if bboxes_type == Type.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == Type.TRAINING:
            bboxes_dict = self._training_bboxes
        for i, (image, bboxes, target_bboxes) in enumerate(zip(bboxes_dict['images'], bboxes_dict['detected_bboxes'], bboxes_dict['gt_bboxes'])):
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

        def _filter_bboxes(bboxes_dict):
            for image_count, (targets, detected) in enumerate(zip(bboxes_dict['gt_bboxes'], bboxes_dict['detected_bboxes'])):
                filtered = [0 for _ in range(len(detected))]
                # Like the AP, the multiple detection of the same object are removed
                # The matched bbox is the one with the high IoU area
                if filter_multiple_detection:
                    for target_bbox in targets:
                        max_iou = -1
                        max_index = -1
                        for i, detected_bbox in enumerate(detected):
                            iou_area = BoundingBox.iou(detected_bbox, target_bbox)
                            if image_count == 1:
                                iou_area = BoundingBox.iou(target_bbox, detected_bbox)
                                print(detected_bbox, target_bbox, iou_area)
                                print(int(detected_bbox._x * 256), int(detected_bbox._y * 256), int(detected_bbox._x2 * 256), int(detected_bbox._y2 * 256))
                                print(int(target_bbox._x * 256), int(target_bbox._y * 256), int(target_bbox._x2 * 256), int(target_bbox._y2 * 256))
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
                                if image_count == 12:
                                    print(iou_area)
                                filtered[i] = 1
                bboxes_dict['filtered'].append(filtered)

        _filter_bboxes(self._training_bboxes)
        _filter_bboxes(self._test_bboxes)

    def add_yolo_bboxes(self, images, targets, preds, bboxes_type: Type):
        if bboxes_type == Type.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == Type.TRAINING:
            bboxes_dict = self._training_bboxes

        imgs_size = images.size()[2:]
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
                print(imgs_size, x, y, w, h)
                gt_boxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x, y, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                    img_size=imgs_size,
                    type_coordinates=CoordinatesType.RELATIVE,
                ))
            bboxes_dict['gt_bboxes'].append(gt_boxes)
            self._img_count += 1


        assert len(preds[0]) <= self._num_bboxes
        imgs_size = images.size()[2:]
        for boxes in preds:
            detected_boxes = []
            for x1, y1, x2, y2, score, label in boxes:
                label, score, = int(label.item()), score.item(),
                x1, y1, x2, y2 = x1.item() , y1.item() , x2.item() , y2.item()
                if label >= 0:
                    detected_boxes.append(BoundingBox(
                        image_name=str(img_count_temp),
                        class_id=str(label),
                        type_coordinates=CoordinatesType.ABSOLUTE,
                        coordinates=(x1, y1, x2 - x1, y2 - y1),
                        bb_type=BBType.DETECTED,
                        format=BBFormat.XYWH,
                        confidence=score,
                        img_size=imgs_size
                    ))
            bboxes_dict['detected_bboxes'].append(detected_boxes)
            img_count_temp += 1