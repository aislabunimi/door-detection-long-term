import time

import numpy as np
import onnx
import onnxruntime
import torch
from onnxconverter_common import float16
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetLoaderBBoxes import DatasetLoaderBBoxes
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.background_grid_network import IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL
from doors_detection_long_term.doors_detector.models.bbox_filter_network_geometric import \
    BboxFilterNetworkGeometricBackground, IMAGE_NETWORK_GEOMETRIC_BACKGROUND

from doors_detection_long_term.doors_detector.models.detr_door_detector import \
    EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_15_EPOCHS_40, DetrDoorDetector
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50, \
    BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_bboxes
from torch.onnx import OperatorExportTypes

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
device = 'cuda'
if __name__ == '__main__':
    print(onnxruntime.get_device())
    grid_dim = [(2**i, 2**i) for i in range(3, 6)][::-1]

    model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=False, grid_network_pretrained=True, dataset_name=FINAL_DOORS_DATASET,
                                                      description=IMAGE_NETWORK_GEOMETRIC_BACKGROUND, description_background=IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL)
    model.to(device)
    model.eval()
    dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name=f'faster_rcnn_general_detector_gibson_dd2_floor1_0.5')
    train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=30, iou_threshold_matching=0.5, apply_transforms_to_train=True, shuffle_boxes=False)

    print(len(train_bboxes), len(test_bboxes))
    test_dataset_bboxes = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = next(iter(test_dataset_bboxes))
    images = images.to(device)
    detected_bboxes = detected_bboxes.to(device)
    for k, v in detected_boxes_grid.items():
        detected_boxes_grid[k] = v.to(device)

    # compute ONNX Runtime output prediction
    times = []
    for i in range(200):
        print(i)
        t = time.time()
        model(images, detected_bboxes, detected_boxes_grid)
        times.append(time.time() - t)
    print(f'FPS: {1/(sum(times[2:]) / (len(times)-2))}')





