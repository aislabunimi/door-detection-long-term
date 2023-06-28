import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_faster_rcnn, seed_everything
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']
epochs_general_detector = [40, 60]
epochs_qualified_detector = [20, 40]
fine_tune_quantity = [15, 25, 50, 75]
device = 'cuda'

seed_everything(seed=0)
def compute_results(model_name, data_loader_test, COLORS, description):
    model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
    model.eval()
    model.to(device)

    evaluator = MyEvaluator()
    evaluator_complete_metric = MyEvaluatorCompleteMetric()

    with torch.no_grad():
        for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test), desc=description):
            images = images.to(device)
            preds = model.model(images)
            preds = [apply_nms(pred, iou_thresh=0.5, confidence_threshold=0.01) for pred in preds]
            for pred in preds:
                pred['labels'] = pred['labels'] - 1
            evaluator.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])
            evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])

    complete_metrics = {}
    metrics = {}
    for iou_threshold in np.arange(0.5, 0.96, 0.05):
        for confidence_threshold in np.arange(0.5, 0.96, 0.05):
            iou_threshold = round(iou_threshold, 2)
            confidence_threshold = round(confidence_threshold, 2)
            metrics[(iou_threshold, confidence_threshold)] = evaluator.get_metrics(iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, door_no_door_task=False, plot_curves=False, colors=COLORS)
            complete_metrics[(iou_threshold, confidence_threshold)] = evaluator_complete_metric.get_metrics(iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, door_no_door_task=False, plot_curves=False, colors=COLORS)

    return metrics, complete_metrics


def save_file(results, complete_results, file_name_1, file_name_2):

    results = np.array(results).T
    columns = ['iou_threshold', 'confidence_threshold', 'house', 'detector', 'epochs_gd', 'epochs_qd', 'label',  'AP', 'total_positives', 'TP', 'FP']
    d = {}
    for i, column in enumerate(columns):
        d[column] = results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../../../results/' + file_name_1) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')

    complete_results = np.array(complete_results).T
    columns = ['iou_threshold', 'confidence_threshold', 'house', 'detector', 'epochs_gd', 'epochs_qd', 'label',  'total_positives', 'TP', 'FP', 'TPm', 'FPm', 'FPiou']
    d = {}
    for i, column in enumerate(columns):
        d[column] = complete_results[i]

    dataframe = pd.DataFrame(d)

    with pd.ExcelWriter('./../../../results/' + file_name_2) as writer:
        if not dataframe.index.name:
            dataframe.index.name = 'Index'
        dataframe.to_excel(writer, sheet_name='s')


model_names_general_detectors = [(globals()[f'EXP_1_{house}_{epochs}_EPOCHS'.upper()], house, epochs) for house in houses for epochs in epochs_general_detector]
model_names_qualified_detectors = [(globals()[f'EXP_2_{house}_EPOCHS_GD_{epochs_general}_EPOCH_QD_{epochs_qualified}_FINE_TUNE_{quantity}'.upper()], house, quantity, epochs_general, epochs_qualified) for house in houses for quantity in fine_tune_quantity for epochs_general in epochs_general_detector for epochs_qualified in epochs_qualified_detector]

results = []
results_complete = []

# General detectors
for model_name, house, epochs in model_names_general_detectors:
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

    metrics, complete_metrics = compute_results(model_name, data_loader_test, COLORS, f'{house} - Epochs GD: {epochs}')

    for (iou_threshold, confidence_threshold), metric in metrics.items():
        for label, values in sorted(metric['per_bbox'].items(), key=lambda v: v[0]):
            results += [[iou_threshold, confidence_threshold, house.replace('_', ''), 'GD', epochs, epochs, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

    for (iou_threshold, confidence_threshold), complete_metric in complete_metrics.items():
        for label, values in sorted(complete_metric.items(), key=lambda v: v[0]):
            results_complete += [[iou_threshold, confidence_threshold, house.replace('_', ''), 'GD', epochs, epochs, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

for model_name, house, quantity, epochs_general, epochs_qualified in model_names_qualified_detectors:
    _, _, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

    metrics, complete_metrics = compute_results(model_name, data_loader_test, COLORS, f'{house} - Epochs GD: {epochs_general} - Epochs qualified {epochs_qualified} - {quantity}%')

    for (iou_threshold, confidence_threshold), metric in metrics.items():
        for label, values in sorted(metric['per_bbox'].items(), key=lambda v: v[0]):
            results += [[iou_threshold, confidence_threshold, house.replace('_', ''), 'QD_' + str(quantity), epochs_general, epochs_qualified, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

    for (iou_threshold, confidence_threshold), complete_metric in complete_metrics.items():
        for label, values in sorted(complete_metric.items(), key=lambda v: v[0]):
            results_complete += [[iou_threshold, confidence_threshold, house.replace('_', ''), 'QD_' + str(quantity), epochs_general, epochs_qualified, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]

save_file(results, results_complete, 'faster_rcnn_ap_simulation.xlsx', 'faster_rcnn_complete_metric_simulation.xlsx')