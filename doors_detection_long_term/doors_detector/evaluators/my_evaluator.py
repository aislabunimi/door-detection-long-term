from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt
from src.bounding_box import BoundingBox
from src.evaluators.pascal_voc_evaluator import calculate_ap_every_point

from doors_detector.evaluators.model_evaluator import ModelEvaluator


class MyEvaluator(ModelEvaluator):

    def get_metrics(self,
                    iou_threshold: float = 0.5,
                    confidence_threshold: float = 0.5,
                    door_no_door_task: bool = False,
                    plot_curves: bool = False,
                    colors = None
        ) -> Dict:
        """
        This method calculates metrics to evaluate a object detection model.
        This metric is similar to the Pascal VOC metric but it is developed specifically for a robotic context.
        In fact, if the model is used by a robot, it has to process a lot of negative images (images without any object to detect).
        To correctly evaluate the model's performance in a robotic context, it is mandatory to consider also the negative images.
        The final goal is to count the TP, FP, FN, calculate precision and recall and calculate AP.
        This metric works as follow:
        1) Bounding boxes are assigned to the image they belong to.
        2) Predicted bounding boxes are divided according to the positiveness of their image.
            The bounding boxes of positive and negative images are processed separately.

        3) Positive images (images with object to detect) are treated similarly to the Pascal VOC metric.
            a) Predicted bounding boxes are filtered by their confidence using confidence_threshold parameter.
                Each bbox with confidence < confidence_threshold is discarded.
            b) For each positive image, the ground truth bboxes are divided according to their class (label)
            c) Now, all predicted bounding boxes (of all positive images) are ordered according to their confidence, in descending mode.
            d) Each predicted bbox is matched with a single ground truth bounding of the same class belonging to the same image.
                A match is composed by a ground truth bbox and a predicted bbox:
                they must have the iou grater than a threshold and this value must be the maximum among all ground truth bbox.
                A True Positive (TP) is a matched predicted bbox, while False Positives (FP) are not matched predicted bboxes.
                A match fails when the iou area is less then the threshold or
                the ground truth bounding box with the grater iou are has already been matched.
                Each ground truth bbox not matched are considered as False Negative (FN)

        4) The bounding boxes belong to the negative images are processed differently:
            a) A new label is introduced: -1. It indicates all negative images' bounding boxes
            b) The negative images are ordered according to the confidence sum of their predicted bboxes
            c) Each negative image is processed to find TP and FP bboxes:
                - a TP is a bounding box with confidence < confidence_threshold (the confidence is too low to be considered a good prediction)
                - a FP is a bounding box with a confidence >= confidence threshold.

        The metric described below refers to bounding box but it can be useful obtain result related to the images.
        In this case, the images are divided in positives and negatives.
        The positive images are processed as follow:
            - the predicted bounding boxes are ordered according to their confidence, in descending mode
            - predicted bounding boxes are matched with the ground truth bboxes
            - a TP is a positive image in which all doors are found (so the are a number of matches >= of the number of ground truth bboxes)
            - a FP is a positive image in which 0 < number of matcher < ground truth bboxes
            - a FN is a positive image with no matches (no doors are found)
        :param iou_threshold:
        :param confidence_threshold:
        :param door_no_door_task:
        :param plot_curves:
        :return:
        """
        gt_bboxes = self.get_gt_bboxes()
        predicted_bboxes = self.get_predicted_bboxes()

        predicted_bboxes_positives = []

        # Labels
        labels = {'-1', '0', '1'}

        # A dictionary containing all bboxes divided by image. DETR produces a fixed number of prediction for every image.
        # A positive images have at least one ground truth bbox, while negatives don't have ground truth
        bboxes_images = {
            box.get_image_name(): {
                'is_positive': False,
                'gt_bboxes': [],
                'predicted_bboxes': [],
                'TP': 0,
                'FP': 0,
            }
            for box in predicted_bboxes}

        # Add ground truth bboxes to each image
        for box in gt_bboxes:

            # If door_no_door_task is true, change all bbox labels in 0
            if door_no_door_task:
                box.set_class_id('0')

            img = bboxes_images[box.get_image_name()]
            img['is_positive'] = True

            # Assign bounding box to its image
            img['gt_bboxes'].append(box)

            labels.add(box.get_class_id())

        # Add predicted bboxes to each image.
        # Divide predicted bounding boxes for the image's type (positive or negative) they belong to.
        # For positive images, bounding boxes with confidence < confidence_threshold are discarded.
        for box in predicted_bboxes:

            if door_no_door_task:
                box.set_class_id('0')

            img = bboxes_images[box.get_image_name()]

            if img['is_positive'] and box.get_confidence() >= confidence_threshold:
                img['predicted_bboxes'].append(box)
                predicted_bboxes_positives.append(box)
            elif not img['is_positive']:
                img['predicted_bboxes'].append(box)

        # Create dictionary to divide TP and FP by label
        result_by_labels = {
            label: {
                'total_positives': sum(1 for box in gt_bboxes if box.get_class_id() == label),
                'TP': [],
                'FP': [],
            } for label in labels
        }

        # Process bounding boxes of positive images

        # For each positive image, ground truth bboxes are divided according their label
        for img in [img for img in bboxes_images.values() if img['is_positive']]:
            d = {}
            for label in labels:
                boxes = [box for box in img['gt_bboxes'] if box.get_class_id() == label]
                d[label] = {
                    'bboxes': boxes,
                    'mask': np.zeros(len(boxes))
                }
            img['gt_bboxes'] = d

        # Order bboxes according to confidence
        predicted_bboxes_positives.sort(key=lambda box: box.get_confidence(), reverse=True)

        for p_box in predicted_bboxes_positives:
            label = p_box.get_class_id()
            img = bboxes_images[p_box.get_image_name()]

            iou_max = float('-inf')
            match_index = -1

            # Find the grater iou area with gt bboxes
            for gt_index, gt_box in enumerate(img['gt_bboxes'][label]['bboxes']):
                iou = BoundingBox.iou(p_box, gt_box)
                if iou > iou_max:
                    iou_max = iou
                    match_index = gt_index

            # If the iou >= threshold_iou and the label is the same, the match is valid
            if iou_max >= iou_threshold and img['gt_bboxes'][label]['mask'][match_index] == 0:
                    # Set gt bbox as matched
                    img['gt_bboxes'][label]['mask'][match_index] = 1

                    # Update image information
                    img['TP'] += 1

                    # Update label information
                    result_by_labels[label]['TP'].append(1)
                    result_by_labels[label]['FP'].append(0)

            # False Positive (if the iou area is less than threshold or the gt box has already been matched)
            else:
                # Update image information
                img['FP'] += 1

                # Update label information
                result_by_labels[label]['TP'].append(0)
                result_by_labels[label]['FP'].append(1)

        # Process negative images
        negative_images = sorted(
            [img for img in bboxes_images.values() if not img['is_positive']],
            key=lambda img: sum(box.get_confidence() for box in img['predicted_bboxes']),
            reverse=False
        )

        for img in negative_images:
            for box in img['predicted_bboxes']:
                result_by_labels['-1']['total_positives'] += 1

                if box.get_confidence() < confidence_threshold:
                    img['TP'] += 1
                    result_by_labels['-1']['TP'].append(1)
                    result_by_labels['-1']['FP'].append(0)
                else:
                    img['FP'] += 1
                    result_by_labels['-1']['TP'].append(0)
                    result_by_labels['-1']['FP'].append(1)




        # Prepare return value
        bboxes_information = {}

        for label, values in result_by_labels.items():
            accumulate_tp = np.cumsum(np.array(values['TP'], dtype=int))
            accumulate_fp = np.cumsum(np.array(values['FP'], dtype=int))
            recall = accumulate_tp / values['total_positives']
            precision = np.divide(accumulate_tp, (accumulate_tp + accumulate_fp))

            [ap, mpre, mrec, _] = calculate_ap_every_point(recall, precision)

            ret = {
                'total_positives': values['total_positives'],
                'TP': np.count_nonzero(values['TP']),
                'FP': np.count_nonzero(values['FP']),
                'precision': precision,
                'recall': recall,
                'AP': ap,
            }

            # The result of labels not presents in the examples are discarded
            if ret['total_positives'] > 0:
                bboxes_information[label] = ret

        if plot_curves:
            plt.close()
            for label, values in sorted(bboxes_information.items(), key=lambda v: v[0]):
                precision = values['precision']
                recall = values['recall']
                p = plt.plot(recall, precision, label=f'{label}')
                if colors is not None:
                    p[0].set_color(colors[int(label)])



            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.xlim([-0.1, 1.1])
            plt.ylim([-0.1, 1.1])
            plt.title('Precision/Recall Curve')
            plt.legend(shadow=True)
            plt.grid()
            plt.show()

        images_information = {
            label: {
                'total_positives': 0,
                'TP': 0,
                'FP': 0,
                'FN': 0
            } for label in ['0', '1']
        }

        for img in bboxes_images.values():
            # Positive images
            if img['is_positive']:
                images_information['1']['total_positives'] += 1
                count_gt_bboxes = sum(len(v['bboxes']) for label, v in img['gt_bboxes'].items())

                if img['TP'] >= count_gt_bboxes:
                    images_information['1']['TP'] += 1
                elif img['TP'] == 0:
                    images_information['1']['FN'] += 1
                else:
                    images_information['1']['FP'] += 1

            # Negative images
            else:
                images_information['0']['total_positives'] += 1
                if img['TP'] == len(img['predicted_bboxes']):
                    images_information['0']['TP'] += 1
                else:
                    images_information['0']['FP'] += 1

        return {'per_bbox': bboxes_information, 'per_image': images_information}
