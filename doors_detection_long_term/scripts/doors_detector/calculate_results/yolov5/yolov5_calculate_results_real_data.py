import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

houses = ['floor1', 'floor4', 'chemistry_floor0']
datasets = ['gibson', 'deep_doors_2', 'gibson_deep_doors_2']
epochs_general_detector = [60]
epochs_qualified_detector = [40]
fine_tune_quantity = [15, 25, 50, 75]
device = 'cuda'


def compute_results(model_name, data_loader_test, description):
    model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
    model.eval()
    model.to(device)

    evaluator = MyEvaluator()
    evaluator_complete_metric = MyEvaluatorCompleteMetric()


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
            evaluator_complete_metric.add_predictions_yolo(targets=targets, predictions=preds, imgs_size=[images.size()[2], images.size()[3]])

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)
    complete_metrics = evaluator_complete_metric.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)

    mAP = 0
    print('Results per bounding box:')
    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        mAP += values['AP']
        print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
        #print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
    #print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')

    return metrics, complete_metrics


def save_file(results, complete_results, file_name_1, file_name_2):

    results = np.array(results).T
    columns = ['house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd', 'label',  'AP', 'total_positives', 'TP', 'FP']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../../../results/' + file_name_1) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')

    complete_results = np.array(complete_results).T
    columns = ['house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd', 'label',  'total_positives', 'TP', 'FP', 'TPm', 'FPm', 'FPiou']
    d = {}
    for i, column in enumerate(columns):
        d[column] = complete_results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../../../results/' + file_name_2) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')


model_names_general_detectors = [(globals()[f'EXP_GENERAL_DETECTOR_{dataset}_{epochs}_EPOCHS'.upper()], dataset, epochs) for dataset in datasets for epochs in epochs_general_detector]
model_names_qualified_detectors = [(globals()[f'EXP_2_{house}_{dataset}_EPOCHS_GD_{epochs_general}_EPOCHS_QD_{epochs_qualified}_FINE_TUNE_{quantity}'.upper()], house, dataset, quantity, epochs_general, epochs_qualified) for house in houses for dataset in datasets for quantity in fine_tune_quantity for epochs_general in epochs_general_detector for epochs_qualified in epochs_qualified_detector]

results = []
results_complete = []

# General detectors
for model_name, dataset, epochs, in model_names_general_detectors:
    print(model_name)

    for house in houses:
        _, test, _, _ = get_final_doors_dataset_real_data(folder_name=house, train_size=0.25)
        data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

        metrics, complete_metrics = compute_results(model_name, data_loader_test, f'Test on {house}, GD trained on {dataset} - Epochs GD: {epochs}')

        for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
            results += [[house, 'GD', dataset, epochs, epochs, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

        for label, values in sorted(complete_metrics.items(), key=lambda v: v[0]):
            results_complete += [[house.replace('_', ''), 'GD', dataset, epochs, epochs, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

for model_name, house, dataset, quantity, epochs_general, epochs_qualified in model_names_qualified_detectors:
    _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name=house, train_size=0.25)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

    metrics, complete_metrics = compute_results(model_name, data_loader_test, f'{house} - GD trained on {dataset} - Epochs GD: {epochs_general} - Epochs qualified {epochs_qualified} - {quantity}%')

    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        results += [[house, f'QD_{quantity}', dataset, epochs_general, epochs_qualified, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]
    for label, values in sorted(complete_metrics.items(), key=lambda v: v[0]):
        results_complete += [[house.replace('_', ''), 'QD_' + str(quantity), dataset, epochs_general, epochs_qualified, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

save_file(results, results_complete, 'yolov5_ap_real_data.xlsx', 'yolov5_complete_metric_real_data.xlsx')