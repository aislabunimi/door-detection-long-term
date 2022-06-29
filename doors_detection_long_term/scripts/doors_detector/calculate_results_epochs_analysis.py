import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.models.detr_door_detector import DetrDoorDetector
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']
epochs_general_detector = [10, 20, 40, 60]
fine_tune_quantity = [25, 50, 75]
device = 'cuda'


def compute_results(model_name, data_loader_test, COLORS):
    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
    model.eval()
    model.to(device)

    evaluator = MyEvaluator()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
        images = images.to(device)
        outputs = model(images)
        evaluator.add_predictions(targets=targets, predictions=outputs)

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.5, door_no_door_task=False, plot_curves=True, colors=COLORS)
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
    columns = ['env', 'epochs', 'label',  'AP', 'total_positives', 'TP', 'FP']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../results/' + file_name) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')


# General detectors
for epoch in epochs_general_detector:
    results = []

    for house in houses:
        model_description = globals()[f'EXP_1_{house}_EPOCHS_ANALYSIS_{epoch}'.upper()]
        _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=True)
        data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

        metrics = compute_results(model_description, data_loader_test, COLORS)

        for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
            results += [[house.replace('_', ''), epoch, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

    save_file(results, 'gd_epochs_analysis_' + str(epoch) + '.xlsx')


# Qualified detectors
for epoch in epochs_general_detector:

    for train_size in [25, 50, 75]:
        results = []
        for house in houses:
            model_description = globals()[f'EXP_2_{house}_{train_size}_EPOCHS_ANALYSIS_{epoch}'.upper()]
            _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=2, folder_name=house.replace('_', ''), train_size=train_size/100, use_negatives=True)
            data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

            metrics = compute_results(model_description, data_loader_test, COLORS)

            for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
                results += [[house.replace('_', ''), epoch, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

        save_file(results, f'qd_{train_size}_epochs_analysis_' + str(epoch) + '.xlsx')