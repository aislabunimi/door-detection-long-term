import cv2
import torch.optim
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 20

dataset_creator_bboxes = DatasetCreatorBBoxes()
dataset_creator_bboxes.set_folder_name('faster_rcnn_general_detector_gibson_deep_doors_2')
houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']

for house in houses:
    train, test, labels, _ = get_final_doors_dataset_bbox_filter_one_house(folder_name=house.replace('_', ''), use_negatives=False)

    data_loader_train = DataLoader(train, batch_size=1, collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)

    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)

    model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=globals()[f'EXP_1_{house}_60_EPOCHS'.upper()],
                       box_score_thresh=0.0, box_nms_thresh=1.0, box_detections_per_img=300)

    model.to('cuda')
    model.model.eval()
    ap_metric_classic = {0: 0, 1: 0}
    complete_metric_classic = {'TP': 0, 'FP': 0, 'BFD': 0}
    with torch.no_grad():
        for images, targets, converted_boxes in tqdm(data_loader_train, total=len(data_loader_train)):
            images = images.to('cuda')
            preds = model.model(images)
            for pred in preds:
                pred['labels'] = pred['labels']-1

            dataset_creator_bboxes.add_faster_rcnn_bboxes(images, targets, preds, ExampleType.TRAINING)

        c = int((3/5) * len(data_loader_test))
        for i, (images, targets, converted_boxes) in tqdm(enumerate(data_loader_test), total=len(data_loader_test)):
            images = images.to('cuda')
            preds = model.model(images)
            for pred in preds:
                pred['labels'] = pred['labels']-1

            dataset_creator_bboxes.add_faster_rcnn_bboxes(images, targets, preds, ExampleType.TEST if i > c else ExampleType.TRAINING)

    dataset_creator_bboxes.export_dataset()

train, validation, labels, _ = get_deep_doors_2_relabelled_dataset_for_gd(fixed_scale=256)
data_loader_train = DataLoader(train, batch_size=1, collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)

data_loader_test = DataLoader(validation, batch_size=1, collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)
model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_GENERAL_DETECTOR_GIBSON_60_EPOCHS,
                   box_score_thresh=0.0, box_nms_thresh=1.0, box_detections_per_img=300)
model.to('cuda')
model.eval()
with torch.no_grad():
    for images, targets, converted_boxes in tqdm(data_loader_train, total=len(data_loader_train)):
        images = images.to('cuda')
        preds = model.model(images)
        for pred in preds:
            pred['labels'] = pred['labels']-1
        dataset_creator_bboxes.add_faster_rcnn_bboxes(images, targets, preds, ExampleType.TRAINING)

    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test)):
        images = images.to('cuda')
        preds = model.model(images)
        for pred in preds:
            pred['labels'] = pred['labels']-1
        dataset_creator_bboxes.add_faster_rcnn_bboxes(images, targets, preds, ExampleType.TEST)

    dataset_creator_bboxes.export_dataset()
