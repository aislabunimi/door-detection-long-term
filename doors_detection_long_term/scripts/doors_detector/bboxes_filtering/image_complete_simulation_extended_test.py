import math
from collections import OrderedDict

import cv2
import numpy as np
import torch.optim
from matplotlib import pyplot as plt
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.ops import FeaturePyramidNetwork
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetLoaderBBoxes import DatasetLoaderBBoxes
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.background_grid_network import IMAGE_GRID_NETWORK, \
    IMAGE_GRID_NETWORK_GIBSON_DD2
from doors_detection_long_term.doors_detector.models.bbox_filter_network_geometric import \
    BboxFilterNetworkGeometricBackground, IMAGE_NETWORK_GEOMETRIC_BACKGROUND, bbox_filtering_nms, \
    BboxFilterNetworkGeometricLabelLoss, BboxFilterNetworkGeometricSuppressLoss, \
    BboxFilterNetworkGeometricConfidenceLoss
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.doors_detector.utilities.util.bboxes_fintering import bounding_box_filtering_yolo, \
    check_bbox_dataset, plot_results, plot_grid_dataset

torch.autograd.detect_anomaly(True)
colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 30

grid_dim = [(2**i, 2**i) for i in range(3, 6)][::-1]

iou_threshold_matching_metric = 0.5
iou_threshold_matching = 0.5
confidence_threshold = 0.75
confidence_threshold_metric = 0.38

def fix_first_layer_of_background_network(n):
    any([x in n for x in ['mask_network', 'fpn.bn1.weight', 'fpn.bn1.bias', 'fpn.layer1', ]])
def fix_background_network(n):
    return any([x in n for x in ['mask_network', 'background_network', 'fpn.bn1.weight', 'fpn.bn1.bias', 'fpn.layer1']])

def fix_only_fpn(n):
    return any([x in n for x in ['mask_network', 'fpn.', 'fpn.bn1.weight', 'fpn.bn1.bias', 'fpn.layer1', ]])

def fix_all_not_background_network(n):
    return any([x in n for x in ['mask_network', 'fpn.bn1.weight', 'fpn.bn1.bias', 'fpn.layer1', ]]) or 'background_network' not in n


training_datasets = [('dataset_complete', 'yolov5_general_detector_gibson_deep_doors_2_door_nodoor'), ('dataset_only_door', 'yolov5_general_detector_gibson_deep_doors_2')]
batch_sizes = [(f'batch_size_{2**i}', 2**i) for i in range(2, 7)]
activate_scheduler = [('scheduler_yes', True), ('scheduler_no', False)]
parameters_grad = [('fix_first_part_backbone', [fix_first_layer_of_background_network for _ in range(60)]),
    ('background_network_fixed', [fix_background_network for _ in range(60)]),
                   ('fix_fpn', [fix_only_fpn for _ in range(60)]),
                   ('fix_background_and_other_alternate'), [fix_background_network if i % 10 < 5 else fix_all_not_background_network for i in range(60)],
                   ]
optimizers = [('SGD_0.01', lambda parameters: optim.SGD(params=parameters, lr=0.01)),
              ('ADAM_0.001', lambda parameters: optim.Adam(params=parameters, lr=0.001)),
              ('ADAMW_0.001', lambda parameters: optim.AdamW(params=parameters, lr=0.001))]

for td, bs, activate_s, pg, opt in [(t, b, a, p, o) for t in training_datasets for b in batch_sizes for a in activate_scheduler for p in parameters_grad for o in optimizers]:
    save_path = f'./image_complete/{td[0]}_{bs[0]}_{activate_s[0]}_{pg[0]}_{opt[0]}'.upper()
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    try:
        dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name=td[1])
        train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)

        print(len(train_bboxes), len(test_bboxes))
        train_dataset_bboxes = DataLoader(train_bboxes, batch_size=bs[1], collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4, shuffle=False)
        test_dataset_bboxes = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)
        #check_bbox_dataset(test_dataset_bboxes, confidence_threshold=confidence_threshold, scale_number=(32, 32))

        # Calculate Metrics in real worlds
        houses = ['floor1', 'floor4', 'chemistry_floor0']

        datasets_real_worlds = {}
        with torch.no_grad():
            for house in houses:
                dataset_loader = DatasetLoaderBBoxes(folder_name='yolov5_general_detector_gibson_dd2_' + house)
                train_bboxes, test_bboxes = dataset_loader.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)
                datasets_real_worlds[house] = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4, shuffle=False)

        #check_bbox_dataset(datasets_real_worlds['floor4'], confidence_threshold, scale_number=(32, 32))

        nms_performance = {'sim_test': {}, 'sim_train': {}}
        nms_performance_ap = {'sim_test': {}, 'sim_train': {}}

        # Calculate results
        evaluator_complete_metric = MyEvaluatorCompleteMetric()
        evaluator_ap = MyEvaluator()
        for data in train_dataset_bboxes:
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
            detected_bboxes = detected_bboxes.transpose(1, 2)
            detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold, iou_threshold=0.5, img_size=images.size()[::-1][:2])
            evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
            evaluator_ap.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
        metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)
        metrics_ap = evaluator_ap.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)

        for label, values in metrics.items():
            for k, v in values.items():
                if k not in nms_performance['sim_train']:
                    nms_performance['sim_train'][k] = v
                else:
                    nms_performance['sim_train'][k] += v

        for label, v in metrics_ap['per_bbox'].items():
            nms_performance_ap['sim_train'][label] = v['AP']

        # Calculate results
        evaluator_complete_metric = MyEvaluatorCompleteMetric()
        evaluator_ap = MyEvaluator()
        for data in test_dataset_bboxes:
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
            detected_bboxes = detected_bboxes.transpose(1, 2)
            detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold, iou_threshold=0.5, img_size=images.size()[::-1][:2])
            evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
            evaluator_ap.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

        metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)
        metrics_ap = evaluator_ap.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)


        for label, values in metrics.items():
            for k, v in values.items():
                if k not in nms_performance['sim_test']:
                    nms_performance['sim_test'][k] = v
                else:
                    nms_performance['sim_test'][k] += v

        for label, v in metrics_ap['per_bbox'].items():
            nms_performance_ap['sim_test'][label] = v['AP']


        for house in houses:
            nms_performance[house] = {}
            nms_performance_ap[house] = {}
            evaluator_complete_metric = MyEvaluatorCompleteMetric()
            evaluator_ap = MyEvaluator()
            for data in datasets_real_worlds[house]:
                images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
                detected_bboxes = detected_bboxes.transpose(1, 2)
                detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold, iou_threshold=0.5, img_size=images.size()[::-1][:2])

                evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
                evaluator_ap.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
            metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)
            metrics_ap = evaluator_ap.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)

            for label, values in metrics.items():
                for k, v in values.items():
                    if k not in nms_performance[house]:
                        nms_performance[house][k] = v
                    else:
                        nms_performance[house][k] += v
            for label, v in metrics_ap['per_bbox'].items():
                nms_performance_ap[house][label] = v['AP']

        # Plots
        for env, values in nms_performance.items():
            fig = plt.figure()
            plt.axhline(values['TP'], label='TP', color='green', linestyle='--')
            plt.axhline(values['FP'], label='FP', color='blue', linestyle='--')
            plt.axhline(values['FPiou'], label='FPiou', color='red', linestyle='--')
            plt.title('env')
            plt.legend()
            plt.savefig(save_path + f'/COMPLETE_METRIC_{env}.svg')

        for env, values in nms_performance_ap.items():
            fig = plt.figure()
            plt.axhline(values['0'], label='Closed', color='red', linestyle='--')
            plt.axhline(values['1'], label='Open', color='green', linestyle='--')
            plt.title('env')
            plt.legend()
            plt.savefig(save_path + f'/AP_{env}.svg')


        bbox_model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=IMAGE_NETWORK_GEOMETRIC_BACKGROUND, description_background=IMAGE_GRID_NETWORK_GIBSON_DD2)
        bbox_model.to('cuda')

        criterion_new_labels = BboxFilterNetworkGeometricLabelLoss()
        criterion_suppress = BboxFilterNetworkGeometricSuppressLoss()
        criterion_confidence = BboxFilterNetworkGeometricConfidenceLoss()
        criterion_new_labels.to('cuda')
        criterion_suppress.to('cuda')
        criterion_confidence.to('cuda')

        optimizer = opt[1](bbox_model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


                #print(n)

        logs = {'train': {'loss_label':[], 'loss_confidence':[], 'loss_final':[], 'loss_suppress':[]},
                'test': {'loss_label':[], 'loss_confidence':[], 'loss_final':[],'loss_suppress':[]},
                'test_real_world': {h:{'loss_label':[], 'loss_confidence':[], 'loss_final':[],'loss_suppress':[]} for h in houses},
                'ap': {0: [], 1: []},
                'complete_metric': {'TP': [], 'FP': [], 'BFD': []}}

        net_performance = {}
        net_performance_ap = {}
        for env, metrics in nms_performance.items():
            net_performance[env] = {}
            for metric, v in metrics.items():
                net_performance[env][metric] = []

        for env, metrics in nms_performance_ap.items():
            net_performance_ap[env] = {}
            for metric, _ in metrics.items():
                net_performance_ap[env][metric] = []

        for epoch in range(60):
            for n, p in bbox_model.named_parameters():
                if pg[1][epoch](n):
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            if activate_s:
                scheduler.step()

            bbox_model.train()
            criterion_new_labels.train()
            criterion_suppress.train()
            criterion_confidence.train()
            optimizer.zero_grad()

            for i, data in tqdm(enumerate(train_dataset_bboxes), total=len(train_dataset_bboxes), desc=f'Training epoch {epoch}'):

                images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
                images = images.to('cuda')

                for k, v in image_grids.items():
                    image_grids[k] = v.to('cuda')

                for k, v in detected_boxes_grid.items():
                    detected_boxes_grid[k] = v.to('cuda')

                detected_bboxes = detected_bboxes.to('cuda')
                confidences = confidences.to('cuda')
                labels_encoded = labels_encoded.to('cuda')
                ious = ious.to('cuda')

                preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
                loss_label = criterion_new_labels(preds[0], labels_encoded)
                loss_suppress = criterion_suppress(preds[1], confidences)
                loss_confidence = criterion_confidence(preds[2], ious)
                final_loss = loss_label + loss_suppress + loss_confidence

                #print(final_loss.item())
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()

            with torch.no_grad():
                bbox_model.eval()
                criterion_new_labels.eval()

                temp_losses_final = {'loss_label':[], 'loss_confidence':[], 'loss_final':[], 'loss_suppress':[]}
                evaluator_complete_metric = MyEvaluatorCompleteMetric()
                evaluator_ap = MyEvaluator()
                for i, data in tqdm(enumerate(train_dataset_bboxes), total=len(train_dataset_bboxes), desc=f'Test on training set epoch {epoch}'):

                    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
                    images = images.to('cuda')
                    for k, v in image_grids.items():
                        image_grids[k] = v.to('cuda')

                    for k, v in detected_boxes_grid.items():
                        detected_boxes_grid[k] = v.to('cuda')

                    detected_bboxes = detected_bboxes.to('cuda')
                    confidences = confidences.to('cuda')
                    labels_encoded = labels_encoded.to('cuda')
                    ious = ious.to('cuda')

                    preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
                    new_labels, new_labels_indexes = torch.max(preds[0].to('cpu'), dim=2, keepdim=False)
                    detected_bboxes = detected_bboxes.transpose(1, 2).to('cpu')

                    # Modify confidences according to the model output

                    new_confidences = preds[2]
                    _, new_confidences_indexes = torch.max(new_confidences, dim=2)
                    new_confidences_indexes = new_confidences_indexes
                    new_confidences_indexes[new_confidences_indexes < 0] = 0
                    new_confidences_indexes[new_confidences_indexes > 9] = 9
                    new_confidences_indexes = new_confidences_indexes * 0.1

                    detected_bboxes[:, :, 4] = new_confidences_indexes

                    # Remove bboxes with background network
                    new_labels_indexes[preds[1] < 0.5] = 0

                    # Filtering bboxes according to new labels
                    detected_bboxes = torch.unbind(detected_bboxes, 0)
                    detected_bboxes = [b[i != 0, :] for b, i in zip(detected_bboxes, new_labels_indexes)]

                    detected_bboxes = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes, preds[0].to('cpu'), new_labels_indexes)]
                    # Delete bboxes according to the background network

                    detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=0.0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
                    evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
                    evaluator_ap.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

                    loss_label = criterion_new_labels(preds[0], labels_encoded)
                    loss_suppress = criterion_suppress(preds[1], confidences)
                    loss_confidence = criterion_confidence(preds[2], ious)
                    final_loss = loss_label + loss_suppress + loss_confidence

                    temp_losses_final['loss_final'].append(final_loss.item())
                    temp_losses_final['loss_label'].append(loss_label.item())
                    temp_losses_final['loss_suppress'].append(loss_suppress.item())
                    temp_losses_final['loss_confidence'].append(loss_confidence.item())

                metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)
                metrics_ap = evaluator_ap.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)

                for label, values in metrics.items():
                    for k, v in values.items():
                        if len(net_performance['sim_train'][k]) == epoch:
                            net_performance['sim_train'][k].append(v)
                        else:
                            net_performance['sim_train'][k][-1] += v

                for label, v in metrics_ap['per_bbox'].items():
                    net_performance_ap['sim_train'][label].append(v['AP'])

                logs['train']['loss_final'].append(sum(temp_losses_final['loss_final']) / len(temp_losses_final['loss_final']))
                logs['train']['loss_suppress'].append(sum(temp_losses_final['loss_suppress']) / len(temp_losses_final['loss_suppress']))
                logs['train']['loss_label'].append(sum(temp_losses_final['loss_label']) / len(temp_losses_final['loss_label']))
                logs['train']['loss_confidence'].append(sum(temp_losses_final['loss_confidence']) / len(temp_losses_final['loss_confidence']))

                temp_losses_final = {'loss_label':[], 'loss_confidence':[], 'loss_final':[], 'loss_suppress':[]}
                evaluator_complete_metric = MyEvaluatorCompleteMetric()
                evaluator_ap = MyEvaluator()

                for i, data in tqdm(enumerate(test_dataset_bboxes), total=len(test_dataset_bboxes), desc=f'TEST epoch {epoch}'):
                    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
                    images = images.to('cuda')
                    for k, v in image_grids.items():
                        image_grids[k] = v.to('cuda')

                    for k, v in detected_boxes_grid.items():
                        detected_boxes_grid[k] = v.to('cuda')

                    detected_bboxes = detected_bboxes.to('cuda')
                    confidences = confidences.to('cuda')
                    labels_encoded = labels_encoded.to('cuda')
                    ious = ious.to('cuda')

                    preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
                    new_labels, new_labels_indexes = torch.max(preds[0].to('cpu'), dim=2, keepdim=False)
                    detected_bboxes = detected_bboxes.transpose(1, 2).to('cpu')

                    # Modify confidences according to the model output
                    # Modify confidences according to the model output
                    new_confidences = preds[2]
                    _, new_confidences_indexes = torch.max(new_confidences, dim=2)
                    new_confidences_indexes = new_confidences_indexes
                    new_confidences_indexes[new_confidences_indexes < 0] = 0
                    new_confidences_indexes[new_confidences_indexes > 9] = 9
                    new_confidences_indexes = new_confidences_indexes * 0.1

                    detected_bboxes[:, :, 4] = new_confidences_indexes

                    # Remove bboxes with background network
                    new_labels_indexes[preds[1] < 0.5] = 0

                    # Filtering bboxes according to new labels
                    detected_bboxes = torch.unbind(detected_bboxes, 0)
                    detected_bboxes = [b[i != 0, :] for b, i in zip(detected_bboxes, new_labels_indexes)]
                    # Modify the label according to the new label assigned by the model
                    detected_bboxes = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes, preds[0].to('cpu'), new_labels_indexes)]

                    detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=0.0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
                    evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
                    evaluator_ap.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

                    loss_label = criterion_new_labels(preds[0], labels_encoded)
                    loss_suppress = criterion_suppress(preds[1], confidences)
                    loss_confidence = criterion_confidence(preds[2], ious)
                    final_loss = loss_label + loss_suppress + loss_confidence

                    temp_losses_final['loss_final'].append(final_loss.item())
                    temp_losses_final['loss_label'].append(loss_label.item())
                    temp_losses_final['loss_suppress'].append(loss_suppress.item())
                    temp_losses_final['loss_confidence'].append(loss_confidence.item())

                    plot_results(epoch=epoch, count=i, env='simulation', images=images, bboxes=detected_bboxes, targets=target_boxes, confidence_threshold=confidence_threshold_metric)

                metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)
                metrics_ap = evaluator_ap.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)

                for label, values in metrics.items():
                    for k, v in values.items():
                        if len(net_performance['sim_test'][k]) == epoch:
                            net_performance['sim_test'][k].append(v)
                        else:
                            net_performance['sim_test'][k][-1] += v

                for label, v in metrics_ap['per_bbox'].items():
                    net_performance_ap['sim_test'][label].append(v['AP'])

                logs['test']['loss_final'].append(sum(temp_losses_final['loss_final']) / len(temp_losses_final['loss_final']))
                logs['test']['loss_suppress'].append(sum(temp_losses_final['loss_suppress']) / len(temp_losses_final['loss_suppress']))
                logs['test']['loss_label'].append(sum(temp_losses_final['loss_label']) / len(temp_losses_final['loss_label']))
                logs['test']['loss_confidence'].append(sum(temp_losses_final['loss_confidence']) / len(temp_losses_final['loss_confidence']))

                # Test with real world data
                for house, dataset_real_world in datasets_real_worlds.items():
                    temp_losses_final = {'loss_label':[], 'loss_confidence':[], 'loss_final':[], 'loss_suppress':[]}
                    temp_accuracy = {0: 0, 1: 0}
                    evaluator_complete_metric = MyEvaluatorCompleteMetric()
                    evaluator_ap = MyEvaluator()

                    for i, data in tqdm(enumerate(dataset_real_world), total=len(dataset_real_world), desc=f'TEST in {house}, epoch {epoch}'):
                        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
                        images = images.to('cuda')
                        for k, v in image_grids.items():
                            image_grids[k] = v.to('cuda')
                        for k, v in detected_boxes_grid.items():
                            detected_boxes_grid[k] = v.to('cuda')
                        detected_bboxes = detected_bboxes.to('cuda')
                        confidences = confidences.to('cuda')
                        labels_encoded = labels_encoded.to('cuda')
                        ious = ious.to('cuda')

                        preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
                        new_labels, new_labels_indexes = torch.max(preds[0].to('cpu'), dim=2, keepdim=False)
                        detected_bboxes = detected_bboxes.transpose(1, 2).to('cpu')

                        # Modify confidences according to the model output
                        new_confidences = preds[2]
                        _, new_confidences_indexes = torch.max(new_confidences, dim=2)
                        new_confidences_indexes = new_confidences_indexes
                        new_confidences_indexes[new_confidences_indexes < 0] = 0
                        new_confidences_indexes[new_confidences_indexes > 9] = 9
                        new_confidences_indexes = new_confidences_indexes * 0.1

                        detected_bboxes[:, :, 4] = new_confidences_indexes

                        # Remove bboxes with background network
                        new_labels_indexes[preds[1] < 0.5] = 0

                        # Filtering bboxes according to new labels
                        detected_bboxes = torch.unbind(detected_bboxes, 0)
                        detected_bboxes = [b[i != 0, :] for b, i in zip(detected_bboxes, new_labels_indexes)]
                        # Modify the label according to the new label assigned by the model
                        detected_bboxes = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes, preds[0].to('cpu'), new_labels_indexes)]

                        detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=.0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
                        evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
                        evaluator_ap.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

                        loss_label = criterion_new_labels(preds[0], labels_encoded)
                        loss_suppress = criterion_suppress(preds[1], confidences)
                        loss_confidence = criterion_confidence(preds[2], ious)
                        final_loss = loss_label + loss_suppress + loss_confidence

                        temp_losses_final['loss_final'].append(final_loss.item())
                        temp_losses_final['loss_label'].append(loss_label.item())
                        temp_losses_final['loss_suppress'].append(loss_suppress.item())
                        temp_losses_final['loss_confidence'].append(loss_confidence.item())

                        plot_results(epoch=epoch, count=i, env=house, images=images, bboxes=detected_bboxes, targets=target_boxes, confidence_threshold=confidence_threshold_metric)

                    logs['test_real_world'][house]['loss_final'].append(sum(temp_losses_final['loss_final']) / len(temp_losses_final['loss_final']))
                    logs['test_real_world'][house]['loss_suppress'].append(sum(temp_losses_final['loss_suppress']) / len(temp_losses_final['loss_suppress']))
                    logs['test_real_world'][house]['loss_label'].append(sum(temp_losses_final['loss_label']) / len(temp_losses_final['loss_label']))
                    logs['test_real_world'][house]['loss_confidence'].append(sum(temp_losses_final['loss_confidence']) / len(temp_losses_final['loss_confidence']))

                    metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)
                    metrics_ap = evaluator_ap.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)

                    for label, values in metrics.items():
                        for k, v in values.items():
                            if len(net_performance[house][k]) == epoch:
                                net_performance[house][k].append(v)
                            else:
                                net_performance[house][k][-1] += v
                    for label, v in metrics_ap['per_bbox'].items():
                        net_performance_ap[house][label].append(v['AP'])

                print(logs['train'], logs['test'])
                #print(net_performance)
                for env, values in net_performance.items():
                    fig = plt.figure()
                    plt.axhline(nms_performance[env]['TP'], label='nms TP', color='green', linestyle='--')
                    plt.axhline(nms_performance[env]['FP'], label='nms FP', color='blue', linestyle='--')
                    plt.axhline(nms_performance[env]['FPiou'], label='nms FPiou', color='red', linestyle='--')

                    plt.plot([i for i in range(len(values['TP']))], values['TP'], label='TP', color='green')
                    plt.plot([i for i in range(len(values['TP']))], values['FP'], label='FP', color='blue')
                    plt.plot([i for i in range(len(values['TP']))], values['FPiou'], label='FPiou', color='red')
                    plt.title('env')
                    plt.legend()
                    plt.savefig(save_path + f'/COMPLETE_METRIC_{env}.svg')
                    plt.close()
                for env, values in net_performance_ap.items():
                    fig = plt.figure()
                    plt.axhline(nms_performance_ap[env]['0'], label='nms Closed', color='red', linestyle='--')
                    plt.axhline(nms_performance_ap[env]['1'], label='nms Open', color='green', linestyle='--')

                    plt.plot([i for i in range(len(values['0']))], values['0'], label='Closed', color='red')
                    plt.plot([i for i in range(len(values['1']))], values['1'], label='Open', color='green')
                    plt.title('env')
                    plt.legend()
                    plt.savefig(save_path + f'/AP_{env}.svg')
                    plt.close()
                for l_type in logs['train'].keys():
                    fig = plt.figure()
                    plt.plot([i for i in range(len(logs['train'][l_type]))], logs['train'][l_type], label='Train loss')
                    plt.plot([i for i in range(len(logs['test'][l_type]))], logs['test'][l_type], label='Test loss')
                    for h in houses:
                        plt.plot([i for i in range(len(logs['test_real_world'][house][l_type]))], logs['test_real_world'][house][l_type], label=f'Loss in {h}')
                    plt.title(f'Losses {l_type}')
                    plt.legend()
                    plt.savefig(save_path + f'/final_losses_{l_type}.svg')
                    plt.close()
            bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})
    except Exception as e:
        print('Eccezione: ', e)










