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
    BboxFilterNetworkGeometricBackground, IMAGE_NETWORK_GEOMETRIC_BACKGROUND
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.doors_detector.utilities.util.bboxes_fintering import bounding_box_filtering_yolo, \
    check_bbox_dataset, plot_results, plot_grid_dataset
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
torch.autograd.detect_anomaly(True)
colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 20

grid_dim = [(2**i, 2**i) for i in range(3, 7)][::-1]

iou_threshold_matching = 0.5
confidence_threshold = 0.75

dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name='yolov5_general_detector_gibson_deep_doors_2')
train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)

print(len(train_bboxes), len(test_bboxes))
train_dataset_bboxes = DataLoader(train_bboxes, batch_size=4, collate_fn=collate_fn_bboxes(use_confidence=False, image_grid_dimensions=grid_dim), num_workers=4, shuffle=False)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=4, collate_fn=collate_fn_bboxes(use_confidence=False, image_grid_dimensions=grid_dim), num_workers=4)
#check_bbox_dataset(train_dataset_bboxes, confidence_threshold=confidence_threshold, scale_number=(8, 8))

# Calculate Metrics in real worlds
houses = ['floor1', 'floor4', 'chemistry_floor0']

datasets_real_worlds = {}
with torch.no_grad():
    for house in houses:
        dataset_loader = DatasetLoaderBBoxes(folder_name='yolov5_general_detector_gibson_dd2_' + house)
        train_bboxes, test_bboxes = dataset_loader.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)
        datasets_real_worlds[house] = DataLoader(test_bboxes, batch_size=4, collate_fn=collate_fn_bboxes(use_confidence=False, image_grid_dimensions=grid_dim), num_workers=4, shuffle=False)

#check_bbox_dataset(datasets_real_worlds['floor4'], confidence_threshold, scale_number=(32, 32))
# Calculate results


bbox_model = BboxFilterNetworkGeometricBackground(initial_channels=6, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=IMAGE_NETWORK_GEOMETRIC_BACKGROUND, description_background=IMAGE_GRID_NETWORK)
bbox_model.to('cuda')

criterion = ImageGridNetworkLoss()

optimizer = optim.Adam(bbox_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion.to('cuda')
#for n, p in bbox_model.named_parameters():
#    if p.requires_grad:
#        print(n)

logs = {'train': {'loss_label':[], 'loss_confidence':[], 'loss_final':[]},
        'test': {'loss_label':[], 'loss_confidence':[], 'loss_final':[]},
        'test_real_world': {'loss_label':[], 'loss_confidence':[], 'loss_final':[]},
        'ap': {0: [], 1: []},
        'complete_metric': {'TP': [], 'FP': [], 'BFD': []}}


train_accuracy = {0: [], 1: []}
test_accuracy = {0: [], 1: []}
real_world_accuracy = {h: {0: [], 1: []} for h in houses}

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
        #detected_bboxes = detected_bboxes.to('cuda')
        #confidences = confidences.to('cuda')
        #labels_encoded = labels_encoded.to('cuda')
        #ious = ious.to('cuda')

        preds = bbox_model(images)
        final_loss = criterion(preds, image_grids, target_boxes_grid)

        #print(final_loss.item())
        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()


    with torch.no_grad():

        train_total = {0:0, 1:0}
        test_total = {0:0, 1:0}
        real_world_total = {h:{0:0, 1:0} for h in houses}

        bbox_model.eval()
        criterion.eval()

        temp_losses_final = []
        temp_accuracy = {0: 0, 1: 0}
        for i, data in tqdm(enumerate(train_dataset_bboxes), total=len(train_dataset_bboxes), desc=f'Training epoch {epoch}'):

            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
            images = images.to('cuda')
            for k, v in image_grids.items():
                image_grids[k] = v.to('cuda')
            #detected_bboxes = detected_bboxes.to('cuda')
            #confidences = confidences.to('cuda')
            #labels_encoded = labels_encoded.to('cuda')
            #ious = ious.to('cuda')

            preds = bbox_model(images)

            final_loss = criterion(preds, image_grids, target_boxes_grid)
            temp_losses_final.append(final_loss.item())

            for grid, gt_grid in zip(preds, image_grids[tuple(preds.size()[1:])]):
                for label in [0,1]:
                    temp_accuracy[label] += torch.count_nonzero(torch.logical_and(grid < 0.5 if label == 0 else grid >= 0.5, gt_grid == label)).item()
                    train_total[label] += torch.count_nonzero(gt_grid == label).item()

        logs['train']['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))
        for label in [0, 1]:
            train_accuracy[label].append(temp_accuracy[label] / train_total[label])

        temp_losses_final = []
        temp_accuracy = {0: 0, 1: 0}
        for i, data in tqdm(enumerate(test_dataset_bboxes), total=len(test_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
            images = images.to('cuda')
            for k, v in image_grids.items():
                image_grids[k] = v.to('cuda')
            #detected_bboxes = detected_bboxes.to('cuda')
            #confidences = confidences.to('cuda')
            #labels_encoded = labels_encoded.to('cuda')
            #ious = ious.to('cuda')

            preds = bbox_model(images)

            final_loss = criterion(preds, image_grids, target_boxes_grid)
            temp_losses_final.append(final_loss.item())

            plot_grid_dataset(epoch=epoch, count=i, env='simulation', images=images, grid_targets=image_grids, target_boxes=target_boxes, preds=preds)

            for grid, gt_grid in zip(preds, image_grids[tuple(preds.size()[1:])]):
                for label in [0,1]:
                    temp_accuracy[label] += torch.count_nonzero(torch.logical_and(grid < 0.5 if label == 0 else grid >= 0.5, gt_grid == label)).item()
                    test_total[label] += torch.count_nonzero(gt_grid == label).item()
        logs['test']['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))
        for label in [0, 1]:
            test_accuracy[label].append(temp_accuracy[label] / test_total[label])

        # Test with real world data
        temp_losses_final = []
        for house, dataset_real_world in datasets_real_worlds.items():
            temp_accuracy = {0: 0, 1: 0}
            for i, data in tqdm(enumerate(dataset_real_world), total=len(dataset_real_world), desc=f'TEST in {house}, epoch {epoch}'):
                images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
                images = images.to('cuda')
                for k, v in image_grids.items():
                    image_grids[k] = v.to('cuda')
                #detected_bboxes = detected_bboxes.to('cuda')
                #confidences = confidences.to('cuda')
                #labels_encoded = labels_encoded.to('cuda')
                #ious = ious.to('cuda')

                preds = bbox_model(images)
                final_loss = criterion(preds, image_grids, target_boxes_grid)
                temp_losses_final.append(final_loss.item())
                for grid, gt_grid in zip(preds, image_grids[tuple(preds.size()[1:])]):
                    for label in [0,1]:
                        temp_accuracy[label] += torch.count_nonzero(torch.logical_and(grid < 0.5 if label == 0 else grid >= 0.5, gt_grid == label)).item()
                        real_world_total[house][label] += torch.count_nonzero(gt_grid == label).item()

                plot_grid_dataset(epoch=epoch, count=i, env=house, images=images, grid_targets=image_grids, target_boxes=target_boxes, preds=preds)

            for label in [0, 1]:
                    real_world_accuracy[house][label].append(temp_accuracy[label] / real_world_total[house][label])
        logs['test_real_world']['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))

    print(logs['train'], logs['test'])

    fig = plt.figure()
    plt.plot([i for i in range(len(logs['train']['loss_final']))], logs['train']['loss_final'], label='Train loss')
    plt.plot([i for i in range(len(logs['train']['loss_final']))], logs['test']['loss_final'], label='Test loss')
    plt.plot([i for i in range(len(logs['train']['loss_final']))], logs['test_real_world']['loss_final'], label='Test loss real world')
    plt.title('Losses')
    plt.legend()
    plt.savefig('image_grid/final_loss.svg')

    fig = plt.figure()
    plt.plot([i for i in range(len(train_accuracy[0]))], train_accuracy[0], label='Accuracy 0')
    plt.plot([i for i in range(len(train_accuracy[1]))], train_accuracy[1], label='Accuracy 1')
    plt.title('Accuracy Train')
    plt.legend()
    plt.savefig('image_grid/accuracy_train.svg')

    fig = plt.figure()
    plt.plot([i for i in range(len(test_accuracy[0]))], test_accuracy[0], label='Accuracy 0')
    plt.plot([i for i in range(len(test_accuracy[1]))], test_accuracy[1], label='Accuracy 1')
    plt.title('Accuracy Test')
    plt.legend()
    plt.savefig('image_grid/accuracy_test.svg')

    for h in houses:
        fig = plt.figure()
        plt.plot([i for i in range(len(real_world_accuracy[h][0]))], real_world_accuracy[h][0], label='Accuracy 0')
        plt.plot([i for i in range(len(real_world_accuracy[h][1]))], real_world_accuracy[h][1], label='Accuracy 1')
        plt.title(f'Accuracy {h}')
        plt.legend()
        plt.savefig(f'image_grid/accuracy_test_{h}.svg')

    bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})










