from abc import abstractmethod
from typing import List, Dict

from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat, CoordinatesType
import torch.nn.functional as F

class ModelEvaluator:
    def __init__(self):
        self._gt_bboxes = []
        self._predicted_bboxes = []

        self._img_count = 0

    def get_gt_bboxes(self) -> List[BoundingBox]:
        """
        Returns a list containing the ground truth bounding boxes
        :return:
        """
        return self._gt_bboxes

    def get_predicted_bboxes(self) -> List[BoundingBox]:
        """
        Returns a list containing the predicted bounding boxes
        :return:
        """
        return self._predicted_bboxes

    def add_predictions(self, targets, predictions, img_size):
        img_count_temp = self._img_count

        for target in targets:
            for label, [x, y, w, h] in zip(target['labels'].tolist(), target['boxes'].tolist()):
                self._gt_bboxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x, y, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                    type_coordinates=CoordinatesType.RELATIVE,
                    img_size=img_size
                ))
            self._img_count += 1

        pred_logits, pred_boxes_images = predictions['pred_logits'], predictions['pred_boxes']
        prob = F.softmax(pred_logits, -1)
        scores_images, labels_images = prob[..., :-1].max(-1)

        for scores, labels, pred_boxes in zip(scores_images, labels_images, pred_boxes_images):
            for score, label, box in zip(scores, labels, pred_boxes):
                label = label.item()
                score = score.item()
                [x, y, w, h] = box.tolist()
                if label >= 0:
                    self._predicted_bboxes.append(
                        BoundingBox(
                            image_name=str(img_count_temp),
                            class_id=str(label),
                            coordinates=(x, y, w, h),
                            bb_type=BBType.DETECTED,
                            format=BBFormat.XYWH,
                            confidence=score,
                            type_coordinates=CoordinatesType.RELATIVE,
                            img_size=img_size
                        )
                    )
            img_count_temp += 1

    def add_predictions_yolo(self, targets, predictions, img_size):
        img_count_temp = self._img_count

        for target in targets:
            for label, [x, y, w, h] in zip(target['labels'].tolist(), target['boxes'].tolist()):
                self._gt_bboxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x, y, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                    type_coordinates=CoordinatesType.RELATIVE,
                    img_size=img_size
                ))
            self._img_count += 1

        for boxes in predictions:
            for x1, y1, x2, y2, score, label in boxes.tolist():
                label = int(label)
                if label >= 0:
                    self._predicted_bboxes.append(
                        BoundingBox(
                            image_name=str(img_count_temp),
                            class_id=str(label),
                            coordinates=(x1, y1, x2 - x1, y2 - y1),
                            bb_type=BBType.DETECTED,
                            format=BBFormat.XYWH,
                            confidence=score
                        )
                    )
            img_count_temp += 1

    def add_predictions_faster_rcnn(self, targets, predictions, imgs_size):
        img_count_temp = self._img_count
        #print(targets, predictions)
        for target in targets:
            for label, [x, y, w, h] in zip(target['labels'].tolist(), target['boxes'].tolist()):
                self._gt_bboxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x - w / 2, y - h / 2, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                ))
            self._img_count += 1

        for prediction in predictions:
            for [x1, y1, x2, y2], label, score in zip(prediction['boxes'].tolist(), prediction['labels'].tolist(), prediction['scores'].tolist()):
                x1, y1, x2, y2 = x1 / imgs_size[0], y1 / imgs_size[1], x2 / imgs_size[0], y2 / imgs_size[1]

                if label >= 0:
                    self._predicted_bboxes.append(
                        BoundingBox(
                            image_name=str(img_count_temp),
                            class_id=str(label),
                            coordinates=(x1, y1, x2 - x1, y2 - y1),
                            bb_type=BBType.DETECTED,
                            format=BBFormat.XYWH,
                            confidence=score
                        )
                    )
            img_count_temp += 1

    @abstractmethod
    def get_metrics(self) -> Dict:
        pass
