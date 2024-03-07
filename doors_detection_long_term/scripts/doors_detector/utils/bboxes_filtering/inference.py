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

device = 'cpu'

house = 'floor1'

boxes = 100
quantity = 0.75

results = []
results_complete = []


dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name=f'faster_rcnn_general_detector_gibson_dd2_{house}_{quantity}')
train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=boxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=False, shuffle_boxes=False)

test_dataset_bboxes = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)
evaluator_complete_metric = MyEvaluatorCompleteMetric()
evaluator_ap = MyEvaluator()
evaluator_complete_metric_tasknet = MyEvaluatorCompleteMetric()
evaluator_ap_tasknet = MyEvaluator()
filter_description = globals()[f'IMAGE_NETWORK_GEOMETRIC_BACKGROUND_GIBSON_DD2_FINE_TUNE_{house}_{int(quantity*100)}_bbox_{boxes}'.upper()]

bbox_model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=False, grid_network_pretrained=False, dataset_name=FINAL_DOORS_DATASET,
                                                  description=filter_description, description_background=IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL)

bbox_model.to(device)
with torch.no_grad():
    bbox_model.eval()
    times = []
    for data in tqdm(test_dataset_bboxes, ):
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data

        # Filternet
        images = images.to(device)
        for k, v in image_grids.items():
            image_grids[k] = v.to(device)

        for k, v in detected_boxes_grid.items():
            detected_boxes_grid[k] = v.to(device)

        detected_bboxes_cuda = detected_bboxes.to(device)

        t = time.time()
        preds = bbox_model(images, detected_bboxes_cuda, detected_boxes_grid)
        #print(preds[0])
        times.append(time.time() - t)
print( f'{1 / np.array(times[2:]).mean()} FPS in {device}')


