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
epochs_general_detector = [40, 60]
epochs_qualified_detector = [20, 40]
fine_tune_quantity = [15, 25, 50, 75]
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

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=True, colors=COLORS)

    return metrics


def save_file(results, file_name):

    results = np.array(results).T
    columns = ['backbone', 'house', 'detector', 'epochs_gd', 'epochs', 'label',  'AP', 'total_positives', 'TP', 'FP']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../results/' + file_name) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')


model_names_general_detectors = [(globals()[f'EXP_1_{house}_2_layers_BACKBONE_{epochs}_EPOCHS'.upper()], house, epochs) for house in houses for epochs in epochs_general_detector]
model_names_qualified_detectors = [(globals()[f'EXP_2_{house}_{quantity}_{epochs_general}_GENERAL_2_layers_BACKBONE_{epochs_qualified}_EPOCHS'.upper()], house, quantity, epochs_general, epochs_qualified) for house in houses for quantity in fine_tune_quantity for epochs_general in epochs_general_detector for epochs_qualified in epochs_qualified_detector]

results = []
# General detectors
for model_name, house, backbone, epochs in model_names_general_detectors:
    print(model_name)
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

    metrics = compute_results(model_name, data_loader_test, COLORS)

    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        results += [[backbone, house.replace('_', ''), 'GD', epochs, epochs, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

for model_name, house, quantity, epochs_general, backbone, epochs_qualified in model_names_qualified_detectors:
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

    metrics = compute_results(model_name, data_loader_test, COLORS)

    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        results += [[backbone, house.replace('_', ''), 'QD_' + str(quantity), epochs_general, epochs_qualified, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

save_file(results, 'epochs_and_backbone_analysis_confidence_75.xlsx')