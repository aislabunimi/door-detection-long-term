import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetLoaderBBoxes import DatasetLoaderBBoxes
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.background_grid_network import *
from doors_detection_long_term.doors_detector.models.bbox_filter_network_geometric import *
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_bboxes
colors = {0: (0, 0, 255), 1: (0, 255, 0)}
torch.autograd.set_detect_anomaly(True)

grid_dim = [(2**i, 2**i) for i in range(3, 6)][::-1]
iou_threshold_matching = 0.5
confidence_threshold_filternet = 0.38
iou_threshold_filternet = 0.5

device = 'cuda'

houses = ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']
quantities = [0.25, 0.50, 0.75]

num_boxes = [10, 30, 50, 100]

results = []
results_complete = []

for house in houses:



    for quantity in quantities:

        for boxes in num_boxes:
            dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name=f'faster_rcnn_general_detector_gibson_dd2_{house}_{quantity}')
            train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=boxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=False, shuffle_boxes=False)

            test_dataset_bboxes = DataLoader(test_bboxes, batch_size=16, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)
            evaluators = {(r1, r2, s): {'AP': MyEvaluator(), 'complete': MyEvaluatorCompleteMetric()}
                          for r1 in range(2) for r2 in range(2) for s in range(2)
            }
            evaluator_complete_metric_tasknet = MyEvaluatorCompleteMetric()
            evaluator_ap_tasknet = MyEvaluator()
            filter_description = globals()[f'IMAGE_NETWORK_GEOMETRIC_BACKGROUND_GIBSON_DD2_FINE_TUNE_{house}_{int(quantity*100)}_bbox_{boxes}'.upper()]

            bbox_model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=True, grid_network_pretrained=False, dataset_name=FINAL_DOORS_DATASET,
                                                              description=filter_description, description_background=IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL)

            bbox_model.to(device)
            with torch.no_grad():
                bbox_model.eval()
                for data in tqdm(test_dataset_bboxes, ):
                    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data

                    # Filternet
                    images = images.to(device)
                    for k, v in image_grids.items():
                        image_grids[k] = v.to(device)

                    for k, v in detected_boxes_grid.items():
                        detected_boxes_grid[k] = v.to(device)

                    detected_bboxes_cuda = detected_bboxes.to(device)

                    preds = bbox_model(images, detected_bboxes_cuda, detected_boxes_grid)
                    #print(preds[0])


                    for rel, res, sup in [(r1, r2, s) for r1 in range(2) for r2 in range(2) for s in range(2)]:
                        _, new_labels_indexes = torch.max(preds[0].to('cpu'), dim=2, keepdim=False)
                        detected_bboxes_predicted = detected_bboxes_cuda.transpose(1, 2).to('cpu')

                        # Modify confidences according to the model output

                        new_confidences = preds[2]
                        _, new_confidences_indexes = torch.max(new_confidences, dim=2)
                        new_confidences_indexes = new_confidences_indexes
                        new_confidences_indexes[new_confidences_indexes < 0] = 0
                        new_confidences_indexes[new_confidences_indexes > 9] = 9
                        new_confidences_indexes = new_confidences_indexes * 0.1

                        if res == 1:
                            detected_bboxes_predicted[:, :, 4] = new_confidences_indexes

                        # Remove bboxes with background network
                        detected_bboxes_predicted = torch.unbind(detected_bboxes_predicted, 0)
                        new_labels_indexes = torch.unbind(new_labels_indexes.to('cpu'), 0)
                        new_labels = torch.unbind(preds[0].to('cpu'), 0)
                        if sup == 1:
                            new_labels_indexes = [new_labels[labels != 0] for labels, new_labels in zip(torch.max(preds[1].to('cpu'), dim=2)[1], new_labels_indexes)]
                            new_labels = [nl[labels != 0, :] for labels, nl in zip(torch.max(preds[1].to('cpu'), dim=2)[1], new_labels)]
                            detected_bboxes_predicted = [bboxes_predicted[labels != 0, :] for labels, bboxes_predicted in zip(torch.max(preds[1].to('cpu'), dim=2)[1], detected_bboxes_predicted)]

                        # Filtering bboxes according to new labels
                        if rel == 1:
                            detected_bboxes_predicted = [b[i != 0, :] for b, i in zip(detected_bboxes_predicted, new_labels_indexes)]
                            detected_bboxes_predicted = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes_predicted, new_labels, new_labels_indexes)]
                            # Delete bboxes according to the background network
                        detected_bboxes_predicted = bbox_filtering_nms(detected_bboxes_predicted, confidence_threshold=0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
                        evaluators[(rel, res, sup)]['complete'].add_predictions_bboxes_filtering(bboxes=detected_bboxes_predicted, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
                        evaluators[(rel, res, sup)]['AP'].add_predictions_bboxes_filtering(bboxes=detected_bboxes_predicted, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

                    # Tasknet
                    detected_bboxes = detected_bboxes.transpose(1, 2)

                    detected_bboxes_tasknet = bbox_filtering_nms(detected_bboxes, confidence_threshold=0.0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
                    evaluator_ap_tasknet.add_predictions_bboxes_filtering(detected_bboxes_tasknet, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
                    evaluator_complete_metric_tasknet.add_predictions_bboxes_filtering(detected_bboxes_tasknet, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])


                confidence_threshold_tasknet = 0.75
                iou_threshold_tasknet = 0.5
                # Calculate metrics
                metrics_tasknet = evaluator_complete_metric_tasknet.get_metrics(confidence_threshold=confidence_threshold_tasknet, iou_threshold=iou_threshold_tasknet)
                metrics_ap_tasknet = evaluator_ap_tasknet.get_metrics(confidence_threshold=confidence_threshold_tasknet, iou_threshold=iou_threshold_tasknet)



                for label, values in sorted(metrics_ap_tasknet['per_bbox'].items(), key=lambda v: v[0]):
                    results += [[0, 0, 0, 'tasknet', house, quantity, boxes, iou_threshold_matching, confidence_threshold_tasknet, iou_threshold_tasknet, confidence_threshold_filternet, iou_threshold_filternet, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]


                for label, values in sorted(metrics_tasknet.items(), key=lambda v: v[0]):
                    results_complete += [[0, 0, 0, 'tasknet', house, quantity, boxes, iou_threshold_matching, confidence_threshold_tasknet, iou_threshold_tasknet, confidence_threshold_filternet, iou_threshold_filternet, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

                for (rel, res, sup), value in evaluators.items():
                    metrics = value['complete'].get_metrics(confidence_threshold=confidence_threshold_filternet, iou_threshold=iou_threshold_filternet)
                    metrics_ap = value['AP'].get_metrics(confidence_threshold=0.38, iou_threshold=0.5)

                    for label, values in sorted(metrics_ap['per_bbox'].items(), key=lambda v: v[0]):
                        results += [[rel, res, sup, 'filternet', house, quantity, boxes, iou_threshold_matching, confidence_threshold_tasknet, iou_threshold_tasknet, confidence_threshold_filternet, iou_threshold_filternet, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]


                    for label, values in sorted(metrics.items(), key=lambda v: v[0]):
                        results_complete += [[rel, res, sup, 'filternet', house, quantity, boxes, iou_threshold_matching, confidence_threshold_tasknet, iou_threshold_tasknet, confidence_threshold_filternet, iou_threshold_filternet, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]




results = np.array(results).T
columns = ['relabeling', 'rescoring', 'suppression', 'model', 'house', 'quantity', 'boxes', 'iou_threshold_matching', 'confidence_threshold_tasknet', 'iou_threshold_tasknet',
           'confidence_threshold_filternet', 'iou_threshold_filternet', 'label',  'AP', 'total_positives', 'TP', 'FP']
d = {}
for i, column in enumerate(columns):
    d[column] = results[i]

dataframe = pd.DataFrame(d)
dataframe.to_csv('./../../../results/filternet_results_ap_ablation.csv', index=False)

complete_results = np.array(results_complete).T
columns = ['relabeling', 'rescoring', 'suppression', 'model', 'house', 'quantity', 'boxes', 'iou_threshold_matching', 'confidence_threshold_tasknet', 'iou_threshold_tasknet',
           'confidence_threshold_filternet', 'iou_threshold_filternet', 'label',  'total_positives', 'TP', 'FP', 'TPm', 'FPm', 'FPiou']
d = {}
for i, column in enumerate(columns):
    d[column] = complete_results[i]

dataframe = pd.DataFrame(d)
dataframe.to_csv('./../../../results/filternet_results_complete_ablation.csv', index=False)