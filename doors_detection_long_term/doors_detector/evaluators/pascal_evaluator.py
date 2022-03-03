from typing import Dict

from src.evaluators.pascal_voc_evaluator import get_pascalvoc_metrics

from doors_detector.evaluators.model_evaluator import ModelEvaluator


class PascalEvaluator(ModelEvaluator):

    def get_metrics(self) -> Dict:
        return get_pascalvoc_metrics(self.get_gt_bboxes(), self.get_predicted_bboxes())