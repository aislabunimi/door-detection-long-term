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
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 20

houses = ['floor1', 'floor4', 'chemistry_floor0']

for house in houses:

    dataset_creator_bboxes = DatasetCreatorBBoxes()
    dataset_creator_bboxes.set_folder_name('yolov5_general_detector_gibson_dd2_' + house)

    train, test, l, _ = get_final_doors_dataset_real_data(folder_name=house, train_size=0.75, transform_train=False)
    labels = l
    data_loader_train = DataLoader(train, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

    model = YOLOv5Model(model_name=YOLOv5, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_GENERAL_DETECTOR_GIBSON_DEEP_DOORS_2_60_EPOCHS)

    model.to('cuda')
    model.model.eval()
    ap_metric_classic = {0: 0, 1: 0}
    complete_metric_classic = {'TP': 0, 'FP': 0, 'BFD': 0}
    with torch.no_grad():
        for images, targets, converted_boxes in tqdm(data_loader_train, total=len(data_loader_train)):
            images = images.to('cuda')
            preds, train_out = model.model(images)

            dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, ExampleType.TRAINING)

        #c = int((3/5) * len(data_loader_test))
        for i, (images, targets, converted_boxes) in tqdm(enumerate(data_loader_test), total=len(data_loader_test)):
            images = images.to('cuda')
            preds, train_out = model.model(images)

            dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, ExampleType.TEST)

    dataset_creator_bboxes.export_dataset()