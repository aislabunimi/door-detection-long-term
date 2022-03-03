import cv2
import numpy as np
import torch
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat

from doors_detection_long_term.doors_detector.baseline.baseline import BaselineMethod
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from dataset_configurator import get_deep_doors_2_labelled_sets

door_no_door_task = True

train, test, labels_set, COLORS = get_deep_doors_2_labelled_sets()
baseline_method = BaselineMethod()
my_evaluator = MyEvaluator()
total_positives = 0
TP = 0
FP = 0
for i, (image, targets, sample) in enumerate(test):
    check = True
    for label in targets['labels']:
        if label > 0:
            check = False
    if not check:
        continue

    detections = baseline_method.get_bounding_boxes(sample.get_bgr_image())
    pred_logits = [[]]
    pred_bboxes = [[]]

    gt_bboxes = []
    for [x, y, w, h] in targets['boxes']:
        total_positives += 1
        gt_bboxes.append(BoundingBox(
            image_name=str(i),
            class_id=str(0),
            coordinates=(x - w / 2, y - h / 2, w, h),
            bb_type=BBType.GROUND_TRUTH,
            format=BBFormat.XYWH,
        ))
    gt_mask = [0 for _ in range(len(gt_bboxes))]

    predicted_bboxes = []
    for detection in detections:
        x, y, w, h = cv2.boundingRect(np.array(detection))
        h_image, w_image = (targets['size'] / 2).tolist()
        x = (x + w / 2) / w_image
        y = (y + h / 2) / h_image
        w /= w_image
        h /= h_image
        predicted_bboxes.append(BoundingBox(
            image_name=str(i),
            class_id=str(0),
            coordinates=(x - w / 2, y - h / 2, w, h),
            bb_type=BBType.DETECTED,
            format=BBFormat.XYWH,
            confidence=1
        ))

    for p_box in predicted_bboxes:
        label = p_box.get_class_id()

        iou_max = float('-inf')
        match_index = -1

        # Find the grater iou area with gt bboxes
        for gt_index, gt_box in enumerate(gt_bboxes):
            iou = BoundingBox.iou(p_box, gt_box)
            if iou > iou_max:
                iou_max = iou
                match_index = gt_index

        # If the iou >= threshold_iou and the label is the same, the match is valid
        if iou_max >= 0.75 and gt_mask[match_index] == 0:
            # Update image information
            TP += 1
            gt_mask[match_index] = 1
        # False Positive (if the iou area is less than threshold or the gt box has already been matched)
        else:
            # Update image information
            FP += 1

        """if TP > 0:
            [h, w, _] = sample.get_bgr_image().shape
            image_small = cv2.resize(sample.get_bgr_image(), (int(round(w / 2)), int(round(h / 2))), interpolation=cv2.INTER_CUBIC)

            for [c1, c2, c3, c4] in detections:
                cv2.polylines(image_small, np.array([[c1, c2, c3, c4]]), True, (0, 255, 0))
        cv2.imshow('ff', image_small)
        cv2.waitKey()"""

print(f'Total_positives = {total_positives}, TP = {TP}, FP = {FP}')
