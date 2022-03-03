from typing import Dict

from src.evaluators.coco_evaluator import get_coco_summary

from doors_detector.evaluators.model_evaluator import ModelEvaluator


class CocoEvaluator(ModelEvaluator):

    def get_metrics(self) -> Dict:
        return get_coco_summary(self.get_gt_bboxes(), self.get_predicted_bboxes())

