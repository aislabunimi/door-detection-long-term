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
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_real_data

houses = ['floor1', 'floor4', 'chemistry_floor0']
general_datasets = ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']
epochs_general_detector = [40, 60]
epochs_qualified_detector = [20, 40]
fine_tune_quantity = [25, 50, 75]
device = 'cuda'


def compute_results(model, data_loader_test):
    model.eval()
    model.to(device)

    evaluator = MyEvaluatorCompleteMetric()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
        images = images.to(device)
        outputs = model(images)
        evaluator.add_predictions(targets=targets, predictions=outputs)

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.5, door_no_door_task=False, plot_curves=False)
    mAP = 0
    print('Results per bounding box:')
    for label, values in sorted(metrics.items(), key=lambda v: v[0]):
        print(f'Label {label} ->\n\tTotal positives = {values["total_positives"]} \n\tTotal detections = {values["total_detections"]} \n\tTP = {values["TP"]} \n\tFP = {values["FP"]} \n\tTPm = {values["TPm"]} \n\tFPm = {values["FPm"]} \n\tFPiou = {values["FPiou"]}')

    return metrics


def save_file(results, file_name):

    results = np.array(results).T
    columns = ['house', 'exp', 'general_dataset', 'epochs_gd', 'epochs_qd', 'label',  'total_positives', 'TP', 'FP', 'TPm', 'FPm', 'FPiou']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../results/' + file_name) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')

results = []

datasets_loaded = {h: DataLoader(get_final_doors_dataset_real_data(folder_name=h, train_size=0.25)[1], batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4) for h in houses}

# General detectors
for dataset, epoch_general in [(d, eg) for d in general_datasets for eg in epochs_general_detector]:
    model_name = globals()[f'EXP_GENERAL_DETECTOR_2_LAYERS_BACKBONE_{dataset}_{epoch_general}_EPOCHS'.upper()]
    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)

    for house in houses:
        data_loader_test = datasets_loaded[house]
        metrics = compute_results(model, data_loader_test)

        for label, values in sorted(metrics.items(), key=lambda v: v[0]):
            results += [[house, 'GD', dataset, epoch_general, epoch_general, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]


for house, dataset, epoch_general, fine_tune, epoch_qualified in [(h, d, eg, f, ef) for h in houses for d in general_datasets for eg in epochs_general_detector for f in fine_tune_quantity for ef in epochs_qualified_detector]:
    model_name = globals()[f'EXP_2_{house}_{dataset}_{epoch_general}_FINE_TUNE_{fine_tune}_EPOCHS_{epoch_qualified}'.upper()]
    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)

    data_loader_test = datasets_loaded[house]
    metrics = compute_results(model, data_loader_test)

    for label, values in sorted(metrics.items(), key=lambda v: v[0]):
        results += [[house, f'QD_{fine_tune}', dataset, epoch_general, epoch_qualified, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

save_file(results, 'results_real_data_metric_complete.xlsx')