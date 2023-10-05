from collections import OrderedDict

import cv2
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
from doors_detection_long_term.doors_detector.models.background_grid_network import IMAGE_GRID_NETWORK
from doors_detection_long_term.doors_detector.models.bbox_filter_network_geometric import \
    BboxFilterNetworkGeometricBackground, IMAGE_NETWORK_GEOMETRIC_BACKGROUND, bbox_filtering_nms, \
    BboxFilterNetworkGeometricLabelLoss
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.doors_detector.utilities.util.bboxes_fintering import bounding_box_filtering_yolo, \
    check_bbox_dataset, plot_results, plot_grid_dataset
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
torch.autograd.detect_anomaly(True)
colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 30

grid_dim = [(2**i, 2**i) for i in range(3, 7)][::-1]

iou_threshold_matching_metric = 0.5
iou_threshold_matching = 0.5
confidence_threshold = 0.75
confidence_threshold_metric = 0.1

dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name='yolov5_general_detector_gibson_deep_doors_2')
train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)

print(len(train_bboxes), len(test_bboxes))
train_dataset_bboxes = DataLoader(train_bboxes, batch_size=16, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4, shuffle=False)
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
# Calculate results
evaluator_complete_metric = MyEvaluatorCompleteMetric()
for data in train_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
    detected_bboxes = detected_bboxes.transpose(1, 2)
    detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold, iou_threshold=0.5, img_size=images.size()[::-1][:2])
    evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)

for label, values in metrics.items():
    for k, v in values.items():
        if k not in nms_performance['sim_train']:
            nms_performance['sim_train'][k] = v
        else:
            nms_performance['sim_train'][k] += v


# Calculate results
evaluator_complete_metric = MyEvaluatorCompleteMetric()
for data in test_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
    detected_bboxes = detected_bboxes.transpose(1, 2)
    detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold, iou_threshold=0.5, img_size=images.size()[::-1][:2])
    evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)


for label, values in metrics.items():
    for k, v in values.items():
        if k not in nms_performance['sim_test']:
            nms_performance['sim_test'][k] = v
        else:
            nms_performance['sim_test'][k] += v

for house in houses:
    nms_performance[house] = {}
    evaluator_complete_metric = MyEvaluatorCompleteMetric()
    for data in datasets_real_worlds[house]:
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
        detected_bboxes = detected_bboxes.transpose(1, 2)
        detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold, iou_threshold=0.5, img_size=images.size()[::-1][:2])

        evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
    metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold, iou_threshold=iou_threshold_matching_metric)
    for label, values in metrics.items():
        for k, v in values.items():
            if k not in nms_performance[house]:
                nms_performance[house][k] = v
            else:
                nms_performance[house][k] += v

# Plots
for env, values in nms_performance.items():
    fig = plt.figure()
    plt.axhline(values['TP'], label='TP', color='green')
    plt.axhline(values['FP'], label='FP', color='blue')
    plt.axhline(values['FPiou'], label='FPiou', color='red')
    plt.title('env')
    plt.legend()
    plt.savefig(f'image_complete/{env}.svg')


bbox_model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=IMAGE_NETWORK_GEOMETRIC_BACKGROUND, description_background=IMAGE_GRID_NETWORK)
bbox_model.to('cuda')


criterion = BboxFilterNetworkGeometricLabelLoss()
criterion.to('cuda')

optimizer = optim.Adam(bbox_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for n, p in bbox_model.named_parameters():
    if any([x in n for x in ['fpn.conv1.weight', 'fpn.bn1.weight', 'fpn.bn1.bias', 'fpn.layer1', 'background_network']]):
        p.requires_grad = False
        #print(n)
# Fix parameters of background network
for n, p in bbox_model.named_parameters():
    #if 'background_network' in n:
    print(n, p.requires_grad)
        #p.requires_grad = True
        #print('IS')

logs = {'train': {'loss_label':[], 'loss_confidence':[], 'loss_final':[]},
        'test': {'loss_label':[], 'loss_confidence':[], 'loss_final':[]},
        'test_real_world': {h:{'loss_label':[], 'loss_confidence':[], 'loss_final':[]} for h in houses},
        'ap': {0: [], 1: []},
        'complete_metric': {'TP': [], 'FP': [], 'BFD': []}}

net_performance = {}
for env, metrics in nms_performance.items():
    net_performance[env] = {}
    for metric, v in metrics.items():
        net_performance[env][metric] = []

for epoch in range(60):
    #scheduler.step()
    bbox_model.train()
    criterion.train()
    optimizer.zero_grad()

    for i, data in tqdm(enumerate(train_dataset_bboxes), total=len(train_dataset_bboxes), desc=f'Training epoch {epoch}'):

        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
        images = images.to('cuda')

        for k, v in image_grids.items():
            image_grids[k] = v.to('cuda')
        detected_bboxes = detected_bboxes.to('cuda')
        #confidences = confidences.to('cuda')
        labels_encoded = labels_encoded.to('cuda')
        #ious = ious.to('cuda')

        preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
        final_loss = criterion(preds, labels_encoded)

        #print(final_loss.item())
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()


    with torch.no_grad():
        bbox_model.eval()
        criterion.eval()

        temp_losses_final = []
        evaluator_complete_metric = MyEvaluatorCompleteMetric()
        for i, data in tqdm(enumerate(train_dataset_bboxes), total=len(train_dataset_bboxes), desc=f'Test on training set epoch {epoch}'):

            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
            images = images.to('cuda')
            for k, v in image_grids.items():
                image_grids[k] = v.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            #confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')
            #ious = ious.to('cuda')

            preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
            new_labels, new_labels_indexes = torch.max(preds[1].to('cpu'), dim=2, keepdim=False)
            detected_bboxes = detected_bboxes.transpose(1, 2).to('cpu')

            # Filtering bboxes according to new labels
            detected_bboxes = torch.unbind(detected_bboxes, 0)
            detected_bboxes = [b[i != 0, :] for b, i in zip(detected_bboxes, new_labels_indexes)]
            # Modify the label according to the new label assigned by the model
            detected_bboxes = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes, preds[1].to('cpu'), new_labels_indexes)]

            detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=0.0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
            evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

            final_loss = criterion(preds, labels_encoded)

            temp_losses_final.append(final_loss.item())

        metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)
        for label, values in metrics.items():
            for k, v in values.items():
                if len(net_performance['sim_train'][k]) == epoch:
                    net_performance['sim_train'][k].append(v)
                else:
                    net_performance['sim_train'][k][-1] += v
        logs['train']['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))

        temp_losses_final = []
        evaluator_complete_metric = MyEvaluatorCompleteMetric()
        for i, data in tqdm(enumerate(test_dataset_bboxes), total=len(test_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
            images = images.to('cuda')
            for k, v in image_grids.items():
                image_grids[k] = v.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            #confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')
            #ious = ious.to('cuda')

            preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
            new_labels, new_labels_indexes = torch.max(preds[1].to('cpu'), dim=2, keepdim=False)
            detected_bboxes = detected_bboxes.transpose(1, 2).to('cpu')

            # Filtering bboxes according to new labels
            detected_bboxes = torch.unbind(detected_bboxes, 0)
            detected_bboxes = [b[i != 0, :] for b, i in zip(detected_bboxes, new_labels_indexes)]
            # Modify the label according to the new label assigned by the model
            detected_bboxes = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes, preds[1].to('cpu'), new_labels_indexes)]

            detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=0.0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
            evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

            final_loss = criterion(preds, labels_encoded)

            temp_losses_final.append(final_loss.item())

            plot_results(epoch=epoch, count=i, env='simulation', images=images, bboxes=detected_bboxes, targets=target_boxes, confidence_threshold=confidence_threshold_metric)

        metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)
        for label, values in metrics.items():
            for k, v in values.items():
                if len(net_performance['sim_test'][k]) == epoch:
                    net_performance['sim_test'][k].append(v)
                else:
                    net_performance['sim_test'][k][-1] += v
            
        logs['test']['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))

        # Test with real world data

        for house, dataset_real_world in datasets_real_worlds.items():
            temp_losses_final = []
            temp_accuracy = {0: 0, 1: 0}
            evaluator_complete_metric = MyEvaluatorCompleteMetric()
            for i, data in tqdm(enumerate(dataset_real_world), total=len(dataset_real_world), desc=f'TEST in {house}, epoch {epoch}'):
                images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
                images = images.to('cuda')
                for k, v in image_grids.items():
                    image_grids[k] = v.to('cuda')
                detected_bboxes = detected_bboxes.to('cuda')
                #confidences = confidences.to('cuda')
                labels_encoded = labels_encoded.to('cuda')
                #ious = ious.to('cuda')

                preds = bbox_model(images, detected_bboxes, detected_boxes_grid)
                new_labels, new_labels_indexes = torch.max(preds[1].to('cpu'), dim=2, keepdim=False)
                detected_bboxes = detected_bboxes.transpose(1, 2).to('cpu')

                # Filtering bboxes according to new labels
                detected_bboxes = torch.unbind(detected_bboxes, 0)
                detected_bboxes = [b[i != 0, :] for b, i in zip(detected_bboxes, new_labels_indexes)]
                # Modify the label according to the new label assigned by the model
                detected_bboxes = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes, preds[1].to('cpu'), new_labels_indexes)]

                detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=.0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
                evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])

                final_loss = criterion(preds, labels_encoded)
                temp_losses_final.append(final_loss.item())

                plot_results(epoch=epoch, count=i, env=house, images=images, bboxes=detected_bboxes, targets=target_boxes, confidence_threshold=confidence_threshold_metric)

            logs['test_real_world'][house]['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))
            metrics = evaluator_complete_metric.get_metrics(confidence_threshold=confidence_threshold_metric, iou_threshold=iou_threshold_matching_metric)
            for label, values in metrics.items():
                for k, v in values.items():
                    if len(net_performance[house][k]) == epoch:
                        net_performance[house][k].append(v)
                    else:
                        net_performance[house][k][-1] += v

        print(logs['train'], logs['test'])
        #print(net_performance)
        for env, values in net_performance.items():
            fig = plt.figure()
            plt.axhline(nms_performance[env]['TP'], label='nms TP', color='green')
            plt.axhline(nms_performance[env]['FP'], label='nms FP', color='blue')
            plt.axhline(nms_performance[env]['FPiou'], label='nms FPiou', color='red')

            plt.plot([i for i in range(len(values['TP']))], values['TP'], label='TP', color='green')
            plt.plot([i for i in range(len(values['TP']))], values['FP'], label='FP', color='blue')
            plt.plot([i for i in range(len(values['TP']))], values['FPiou'], label='FPiou', color='red')
            plt.title('env')
            plt.legend()
            plt.savefig(f'image_complete/{env}.svg')

        fig = plt.figure()
        plt.plot([i for i in range(len(logs['train']['loss_final']))], logs['train']['loss_final'], label='Train loss')
        plt.plot([i for i in range(len(logs['train']['loss_final']))], logs['test']['loss_final'], label='Test loss')
        for h in houses:
            plt.plot([i for i in range(len(logs['test_real_world'][house]['loss_final']))], logs['test_real_world'][house]['loss_final'], label=f'Loss in {h}')
        plt.title('Losses')
        plt.legend()
        plt.savefig('image_complete/final_losses.svg')

    bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})










