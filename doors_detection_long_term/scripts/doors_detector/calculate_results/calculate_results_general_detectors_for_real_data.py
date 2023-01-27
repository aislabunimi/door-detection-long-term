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
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_real_data

envs = ['floor1', 'floor4',]# 'chemistry_floor0']


device = 'cuda'


def compute_results(model, data_loader_test, COLORS):

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
    columns = ['env_test', 'backbone', 'train_dataset', 'epochs_gd', 'label',  'AP', 'total_positives', 'TP', 'FP']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../results/' + file_name) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')

results = []
# General detectors
for backbone in ['FIXED', '2_LAYERS']:
    for train_dataset in ['GIBSON', 'DEEP_DOORS_2', 'GIBSON_DEEP_DOORS_2_HALF', 'GIBSON_DEEP_DOORS_2']:
        epochs_general_detector = [40, 60]
        if backbone == '2_LAYERS':
            epochs_general_detector += [80, 100]

        for epoch_general in epochs_general_detector:
            model_name = globals()[f'EXP_GENERAL_DETECTOR_{backbone}_BACKBONE_{train_dataset}_{epoch_general}_EPOCHS'.upper()]
            print(model_name)

            model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
            model.eval()
            model.to(device)

            for env in envs:
                _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name=env, train_size=0.25)
                data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

                metrics = compute_results(model, data_loader_test, COLORS)

                for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
                    results += [[env, backbone, train_dataset, epoch_general, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]


save_file(results, 'general_detectors_with_real_data.xlsx')