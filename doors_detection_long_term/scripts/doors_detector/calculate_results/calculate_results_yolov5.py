import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn, collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import torchvision.transforms as T


#train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.25, use_negatives=False)
train, test, _, _ = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')
data_loader_train = DataLoader(train, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)
#data_loader_validation = DataLoader(validation, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50)

model.to('cpu')
model.eval()
model.model.eval()
COLORS = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

evaluator = MyEvaluator()
print(len(data_loader_test))

with torch.no_grad():
    for d, data in tqdm(enumerate(data_loader_test), total=len(data_loader_test)):
        images, targets, converted_boxes = data

        #output = model(images)
        preds, train_out = model(images)
        #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
        preds = non_max_suppression(preds,
                                    0.75,
                                    0.45,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=300)

        evaluator.add_predictions_yolo(targets=targets, predictions=preds, imgs_size=[images.size()[2], images.size()[3]])

metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.5, door_no_door_task=False, plot_curves=True, colors=COLORS)
mAP = 0
print('Results per bounding box:')
for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
    mAP += values['AP']
    print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
    print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')
