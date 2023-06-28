from typing import Dict

from src.bounding_box import BoundingBox

from doors_detection_long_term.doors_detector.evaluators.model_evaluator import ModelEvaluator


class MyEvaluatorCompleteMetric(ModelEvaluator):

    def get_metrics(self,
                    iou_threshold: float = 0.5,
                    confidence_threshold: float = 0.5,
                    door_no_door_task: bool = False,
                    plot_curves: bool = False,
                    colors=None
        ) -> Dict:

        gt_bboxes = self.get_gt_bboxes()
        predicted_bboxes = self.get_predicted_bboxes()
        predicted_bboxes = list(filter(lambda b: b.get_confidence() >= confidence_threshold, predicted_bboxes))
        predicted_bboxes.sort(key=lambda box: box.get_confidence(), reverse=True)

        labels = {'0', '1'}

        result_by_labels = {
            label: {
                'total_positives': sum(1 for box in gt_bboxes if box.get_class_id() == label),
                'total_detections': 0,
                'TP': 0,
                'FP': 0,
                'TPm': 0,
                'FPm': 0,
                'FPiou': 0
            } for label in labels
        }
        bboxes_images = {}

        # Divide bounding boxes from images
        for gt_bbox in gt_bboxes:
            img_name = gt_bbox.get_image_name()

            if img_name not in bboxes_images:
                bboxes_images[img_name] = {'bboxes': [], 'mask': []}

            bboxes_images[img_name]['bboxes'].append(gt_bbox)
            bboxes_images[img_name]['mask'].append(0)


        for p_box in predicted_bboxes:
            label = p_box.get_class_id()
            img = bboxes_images[p_box.get_image_name()]

            iou_max = float('-inf')
            match_index = -1

            # Find the grater iou area with gt bboxes
            for gt_index, gt_box in enumerate(img['bboxes']):
                iou = BoundingBox.iou(p_box, gt_box)
                if iou > iou_max:
                    iou_max = iou
                    match_index = gt_index

            result_by_labels[label]['total_detections'] += 1
            # Match completed
            if iou_max >= iou_threshold:
                gt_box = img['bboxes'][match_index]
                # If the bbox has not already been matched
                if img['mask'][match_index] == 0:
                    img['mask'][match_index] = 1

                    if p_box.get_class_id() == gt_box.get_class_id():
                        result_by_labels[label]['TP'] += 1
                    else:
                        result_by_labels[label]['FP'] += 1

                # If the gt_bbox has already been matched
                elif img['mask'][match_index] == 1:
                    if p_box.get_class_id() == gt_box.get_class_id():
                        result_by_labels[label]['TPm'] += 1
                    else:
                        result_by_labels[label]['FPm'] += 1

            else:
                result_by_labels[label]['FPiou'] += 1

        return result_by_labels
