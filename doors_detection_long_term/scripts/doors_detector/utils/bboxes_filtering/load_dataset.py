import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetCreatorBBoxes
from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetLoaderBBoxes import DatasetLoaderBBoxes
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_bboxes
from doors_detection_long_term.doors_detector.utilities.util.bboxes_fintering import check_bbox_dataset

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 20
iou_threshold_matching = 0.5
grid_dim = [(2**i, 2**i) for i in range(3, 7)][::-1]


dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name='yolov5_general_detector_gibson_deep_doors_2')
train, test = dataset_loader_bboxes.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)

train_dataset_bboxes = DataLoader(train, batch_size=2, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4, shuffle=True)
test_dataset_bboxes = DataLoader(test, batch_size=2, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)

check_bbox_dataset(train_dataset_bboxes, confidence_threshold=0.75, scale_number=(32, 32))
