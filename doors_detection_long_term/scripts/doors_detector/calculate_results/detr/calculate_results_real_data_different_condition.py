import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.models.detr_door_detector import DetrDoorDetector
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import \
    get_final_doors_dataset_epoch_analysis, get_final_doors_dataset_real_data

houses = ['floor1', 'floor4',]# 'chemistry_floor0']
epochs_general_detector = [40, 60]
epochs_qualified_detector = [20, 40]
fine_tune_quantity = [15, 25, 50, 75]
datasets = ['GIBSON', 'DEEP_DOORS_2', 'GIBSON_DEEP_DOORS_2']
device = 'cuda'


def compute_results(model_name, data_loader_test, COLORS):
    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
    model.eval()
    model.to(device)

    evaluator = MyEvaluator()
    evaluator_complete_metric =  MyEvaluatorCompleteMetric()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
        images = images.to(device)
        outputs = model(images)
        evaluator.add_predictions(targets=targets, predictions=outputs)
        evaluator_complete_metric.add_predictions(targets=targets, predictions=outputs)

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False, colors=COLORS)
    complete_metrics = evaluator_complete_metric.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False, colors=COLORS)
    mAP = 0
    print('Results per bounding box:')
    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        mAP += values['AP']
        print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
        print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
    print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')

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


results = []
results_complete = []


# General detectors
for house, dataset, epochs_gd in [(h, d, e) for h in houses for d in datasets for e in epochs_general_detector]:

    _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name=house + '_evening', train_size=0.25)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

    model_name = globals()[f'EXP_GENERAL_DETECTOR_2_layers_BACKBONE_{dataset}_{epochs_gd}_EPOCHS'.upper()]

    metrics, complete_metrics = compute_results(model_name, data_loader_test, COLORS)

    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        results += [[house.replace('_', ''), 'GD', dataset, epochs_gd, epochs_gd, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

    for label, values in sorted(complete_metrics.items(), key=lambda v: v[0]):
        results += [[house.replace('_', ''), 'GD', dataset, epochs_gd, epochs_gd, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]


for house, dataset, epochs_gd, epochs_qd, fine_tune in [(h, d, e, eq, ft) for h in houses for d in datasets for e in [60] for eq in epochs_qualified_detector for ft in  fine_tune_quantity]:
    _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name=house + '_evening', train_size=0.25)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

    model_name = globals()[f'EXP_2_{house}_{dataset}_{epochs_gd}_FINE_TUNE_{fine_tune}_EPOCHS_{epochs_qd}'.upper()]

    metrics, complete_metrics = compute_results(model_name, data_loader_test, COLORS)

    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        results += [[house.replace('_', ''), 'QD_' + str(fine_tune_quantity), dataset, epochs_gd, epochs_qd, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

    for label, values in sorted(complete_metrics.items(), key=lambda v: v[0]):
        results += [[house.replace('_', ''), 'QD_' + str(fine_tune_quantity), dataset, epochs_gd, epochs_qd, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

save_file(results, results_complete, 'detr_ap_real_data_different_conditions.xlsx', 'detr_complete_metrics_real_data_different_conditions.xlsx')