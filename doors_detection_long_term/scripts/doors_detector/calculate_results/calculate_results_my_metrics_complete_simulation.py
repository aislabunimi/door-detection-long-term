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
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']
epochs_general_detector = [40, 60]
epochs_qualified_detector = [20, 40]
fine_tune_quantity = [25, 50, 75]
device = 'cuda'


def compute_results(model_name, data_loader_test, COLORS):
    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
    model.eval()
    model.to(device)

    evaluator = MyEvaluatorCompleteMetric()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
        images = images.to(device)
        outputs = model(images)
        evaluator.add_predictions(targets=targets, predictions=outputs)

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=True, colors=COLORS)
    mAP = 0
    print('Results per bounding box:')
    #print(metrics)
    for label, values in sorted(metrics.items(), key=lambda v: v[0]):
        print(f'Label {label} ->\n\tTotal positives = {values["total_positives"]} \n\tTotal detections = {values["total_detections"]} \n\tTP = {values["TP"]} \n\tFP = {values["FP"]} \n\tTPm = {values["TPm"]} \n\tFPm = {values["FPm"]} \n\tFPiou = {values["FPiou"]}')

    return metrics


def save_file(results, file_name):

    results = np.array(results).T
    columns = ['backbone', 'house', 'detector', 'epochs_gd', 'epochs', 'label', 'total_positives', 'TP', 'FP', 'TPm', 'FPm', 'FPiou']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../results/' + file_name) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')


model_names_general_detectors = [(globals()[f'EXP_1_{house}_{backbone}_BACKBONE_{epochs}_EPOCHS'.upper()], house, backbone, epochs) for house in houses for backbone in ['fixed', '2_layers'] for epochs in epochs_general_detector]
model_names_qualified_detectors = [(globals()[f'EXP_2_{house}_{quantity}_{epochs_general}_GENERAL_{backbone}_BACKBONE_{epochs_qualified}_EPOCHS'.upper()], house, quantity, epochs_general, backbone, epochs_qualified) for house in houses for quantity in fine_tune_quantity for epochs_general in epochs_general_detector for backbone in ['fixed', '2_layers'] for epochs_qualified in epochs_qualified_detector]

results = []
# General detectors
for model_name, house, backbone, epochs in model_names_general_detectors:
    print(model_name)
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

    metrics = compute_results(model_name, data_loader_test, COLORS)
    print(metrics)
    for label, values in sorted(metrics.items(), key=lambda v: v[0]):
        results += [[backbone, house.replace('_', ''), 'GD', epochs, epochs, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

for model_name, house, quantity, epochs_general, backbone, epochs_qualified in model_names_qualified_detectors:
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

    metrics = compute_results(model_name, data_loader_test, COLORS)

    for label, values in sorted(metrics.items(), key=lambda v: v[0]):
        results += [[backbone, house.replace('_', ''), 'QD_' + str(quantity), epochs_general, epochs_qualified, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

save_file(results, 'epochs_and_backbone_analysis_my_metric_complete_confidence_75.xlsx')