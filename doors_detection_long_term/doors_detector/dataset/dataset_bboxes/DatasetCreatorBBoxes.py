from enum import Enum

import cv2
import numpy as np
import torch
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat, CoordinatesType
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.torch_dataset import TorchDatasetBBoxes, TRAIN_SET, TEST_SET
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import xywh2xyxy


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
            'bboxes_matched': [],
        }
        self._test_bboxes = {
            'images': [],
            'gt_bboxes': [],
            'bboxes': [],
            'bboxes_matched': [],
        }

    def visualize_bboxes(self, show_filtered: bool = False, bboxes_type: Type = Type.TRAINING):
        if bboxes_type == Type.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == Type.TRAINING:
            bboxes_dict = self._training_bboxes
        for i, (image, bboxes, target_bboxes, matched_bboxes) in enumerate(zip(bboxes_dict['images'], bboxes_dict['bboxes'], bboxes_dict['gt_bboxes'], bboxes_dict['bboxes_matched'])):
            img_size = image.shape
            show_image = image.copy()
            target_image = image.copy()
            background_image = image.copy()
            matched_images = []

            for bbox in target_bboxes:
                x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
                target_image = cv2.rectangle(target_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(bbox.get_class_id())], 2)

                # Create a matched image for each gt bbox
                matched_image = image.copy()
                matched_image = cv2.rectangle(matched_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

                for detected_box, gt_box in matched_bboxes:
                    if gt_box == bbox:
                        x1, y1, x2, y2 = detected_box.get_absolute_bounding_box(BBFormat.XYX2Y2)
                        matched_image = cv2.rectangle(matched_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(bbox.get_class_id())], 2)
                matched_images.append(matched_image)

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
                show_image = cv2.rectangle(show_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(bbox.get_class_id())], 2)

            for detected_box, gt_box in matched_bboxes:
                if gt_box == None:
                    x1, y1, x2, y2 = detected_box.get_absolute_bounding_box(BBFormat.XYX2Y2)
                    background_image = cv2.rectangle(background_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

            show_image = cv2.hconcat([target_image, show_image] + matched_images + [background_image])
            cv2.imshow('show', show_image)
            cv2.waitKey()

    def _match_detected_bboxes(self, dataset_dict, iou_threshold_matching):
        matching_images = []
        for detected_bboxes_image, gt_bboxes_image in zip(dataset_dict['bboxes'], dataset_dict['gt_bboxes']):
            matching = []
            for detected_bbox in detected_bboxes_image:
                matched_gt = None
                match_iou = -1
                for gt_bbox in gt_bboxes_image:
                    iou = BoundingBox.iou(detected_bbox, gt_bbox)
                    if iou >= match_iou and iou >= iou_threshold_matching:
                        matched_gt = gt_bbox
                        match_iou = iou
                matching.append((detected_bbox, matched_gt))
            matching_images.append(matching)
        dataset_dict['bboxes_matched'] = matching_images

    def match_bboxes_with_gt(self, iou_threshold_matching: float = 0.5):
        self._match_detected_bboxes(self._training_bboxes, iou_threshold_matching)
        self._match_detected_bboxes(self._test_bboxes, iou_threshold_matching)

    def create_datasets(self, random_state: int = 42):
        return (TorchDatasetBBoxes(bboxes_dict=self._training_bboxes, set_type=TRAIN_SET),
                TorchDatasetBBoxes(bboxes_dict=self._test_bboxes, set_type=TEST_SET))


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

        for bboxes in preds:
            coords = xywh2xyxy(bboxes[:, :4])
            conf = bboxes[:, 5:] * bboxes[:, 4:5]
            conf, labels = conf.max(1, keepdim=True)
            conf = torch.squeeze(conf)
            labels = torch.squeeze(labels)
            detected_boxes = []
            for (x1, y1, x2, y2), score, label in zip(coords.tolist(), conf.tolist(), labels.tolist()):
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
            detected_boxes = sorted(detected_boxes, key=lambda x: x.get_confidence(), reverse=True)[:self._num_bboxes]
            bboxes_dict['bboxes'].append(detected_boxes)
            img_count_temp += 1