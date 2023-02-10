import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']
epochs_general_detector = [40, 60]
epochs_qualified_detector = [20, 40]
fine_tune_quantity = [25, 50, 75]
device = 'cuda'


def compute_results(model_name, data_loader_test, COLORS, description):
    model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
    model.eval()
    model.to(device)

    evaluator = MyEvaluator()

    with torch.no_grad():
        for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test), desc=description):
            images = images.to(device)
            preds, train_out = model.model(images)
            #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
            preds = non_max_suppression(preds,
                                        0.75,
                                        0.45,

                                        multi_label=True,
                                        agnostic=True,
                                        max_det=300)
            evaluator.add_predictions_yolo(targets=targets, predictions=preds, imgs_size=[images.size()[2], images.size()[3]])

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=True, colors=COLORS)
    mAP = 0
    print('Results per bounding box:')
    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        mAP += values['AP']
        print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
        print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
    print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')

    return metrics


def save_file(results, file_name):

    results = np.array(results).T
    columns = ['house', 'detector', 'epochs_gd', 'epochs', 'label',  'AP', 'total_positives', 'TP', 'FP']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../../results/' + file_name) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')


model_names_general_detectors = [(globals()[f'EXP_1_{house}_{epochs}_EPOCHS'.upper()], house, epochs) for house in houses for epochs in epochs_general_detector]
model_names_qualified_detectors = [(globals()[f'EXP_2_{house}_EPOCHS_GD_{epochs_general}_EPOCH_QD_{epochs_qualified}_FINE_TUNE_{quantity}'.upper()], house, quantity, epochs_general, epochs_qualified) for house in houses for quantity in fine_tune_quantity for epochs_general in epochs_general_detector for epochs_qualified in epochs_qualified_detector]

results = []
# General detectors
for model_name, house, epochs in model_names_general_detectors:
    print(model_name)
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

    metrics = compute_results(model_name, data_loader_test, COLORS, f'{house} - Epochs GD: {epochs}')

    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        results += [[house.replace('_', ''), 'GD', epochs, epochs, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

for model_name, house, quantity, epochs_general, epochs_qualified in model_names_qualified_detectors:
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

    metrics = compute_results(model_name, data_loader_test, COLORS, f'{house} - Epochs GD: {epochs_general} - Epochs qualified {epochs_qualified} - {quantity}%')

    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        results += [[house.replace('_', ''), 'QD_' + str(quantity), epochs_general, epochs_qualified, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

save_file(results, 'yolo_v5_epochs_analysis.xlsx')