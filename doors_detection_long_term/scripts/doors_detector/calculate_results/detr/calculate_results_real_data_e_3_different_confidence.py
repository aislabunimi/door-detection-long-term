import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.models.detr_door_detector import DetrDoorDetector
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn, seed_everything
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import \
    get_final_doors_dataset_epoch_analysis, get_final_doors_dataset_real_data

houses = ['chemistry_floor0']
epochs_general_detector = [60]
epochs_qualified_detector = [40]
fine_tune_quantity = [25]
datasets = ['GIBSON']
device = 'cuda'
seed_everything(seed=0)
iou_threshold = 0.5
confidence_thresholds = [i/100 for i in range(50, 100, 5)]
print(confidence_thresholds)
TPs = []
FPs = []
FPious = []

for confidence_threshold in confidence_thresholds:
    def compute_results(model_name, data_loader_test, COLORS):
        model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
        model.eval()
        model.to(device)

        evaluator = MyEvaluator()
        evaluator_complete_metric =  MyEvaluatorCompleteMetric()

        for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
            images = images.to(device)
            outputs = model(images)
            evaluator.add_predictions(targets=targets, predictions=outputs, img_size=images.size()[2:][::-1])
            evaluator_complete_metric.add_predictions(targets=targets, predictions=outputs, img_size=images.size()[2:][::-1])

        metrics = evaluator.get_metrics(iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, door_no_door_task=False, plot_curves=False, colors=COLORS)
        complete_metrics = evaluator_complete_metric.get_metrics(iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, door_no_door_task=False, plot_curves=False, colors=COLORS)
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

        _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name=house, train_size=0.25)
        data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

        model_name = globals()[f'EXP_GENERAL_DETECTOR_2_layers_BACKBONE_{dataset}_{epochs_gd}_EPOCHS'.upper()]

        metrics, complete_metrics = compute_results(model_name, data_loader_test, COLORS)

        for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
            results += [[house.replace('_', ''), 'GD', dataset, epochs_gd, epochs_gd, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

        for label, values in sorted(complete_metrics.items(), key=lambda v: v[0]):
            results_complete += [[house.replace('_', ''), 'GD', dataset, epochs_gd, epochs_gd, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]


    for house, dataset, epochs_gd, epochs_qd, fine_tune in [(h, d, e, eq, ft) for h in houses for d in datasets for e in [60] for eq in epochs_qualified_detector for ft in  fine_tune_quantity]:
        _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name=house, train_size=0.25)
        data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)

        model_name = globals()[f'EXP_2_{house}_{dataset}_{epochs_gd}_FINE_TUNE_{fine_tune}_EPOCHS_{epochs_qd}'.upper()]

        metrics, complete_metrics = compute_results(model_name, data_loader_test, COLORS)

        for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
            results += [[house.replace('_', ''), 'QD_' + str(fine_tune), dataset, epochs_gd, epochs_qd, label, values['AP'], values['total_positives'], values['TP'], values['FP']]]

        TPs.append(0)
        FPs.append(0)
        FPious.append(0)
        total = 0
        for label, values in sorted(complete_metrics.items(), key=lambda v: v[0]):
            TPs[-1] += values['TP']
            FPs[-1] += values['FP']
            FPious[-1] += values['FPiou']
            total += values['total_positives']
            results_complete += [[house.replace('_', ''), 'QD_' + str(fine_tune), dataset, epochs_gd, epochs_qd, label, values['total_positives'], values['TP'], values['FP'], values['TPm'], values['FPm'], values['FPiou']]]
        TPs[-1] /= (total / 100)
        FPs[-1] /= (total / 100)
        FPious[-1] /= total / 100
print(TPs, FPs, FPious)
fig, ax = subplots(figsize=(10, 5))
print(TPs)
ax.plot(range(len(TPs)), TPs, 'g^-', label='$TP_{\%}$')
ax.plot(range(len(FPs)), FPs, 'b^-', label='$FP_{\%}$')
ax.plot(range(len(FPious)), FPious, 'r^-', label='$BFD_{\%}$')
ax.set_xticks([i for i in range(len(confidence_thresholds))])
ax.set_xticklabels([str(c) for c in  confidence_thresholds], fontsize=15)
ax.set_xlabel('$\\rho_{c}$', fontsize=15)
plt.yticks(fontsize=15)
ax.legend(loc='upper right', fontsize=15)
fig.tight_layout()
plt.show()
    #save_file(results, results_complete, f'detr_ap_real_data_{str(iou_threshold)}.xlsx', f'detr_complete_metrics_real_data_{str(iou_threshold)}.xlsx')