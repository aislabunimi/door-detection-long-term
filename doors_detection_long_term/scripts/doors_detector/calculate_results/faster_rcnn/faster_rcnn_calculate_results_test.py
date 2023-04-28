import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_real_data

device ='cuda'
def compute_results(model_name, data_loader_test, description):
    model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model_name)
    model.eval()
    model.to(device)

    evaluator = MyEvaluator()
    evaluator_complete_metric = MyEvaluatorCompleteMetric()


    with torch.no_grad():
        for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test), desc=description):
            images = images.to(device)
            preds = model.model(images)
            #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
            preds = apply_nms(preds)
            evaluator.add_predictions_faster_rcnn(targets=targets, predictions=preds, imgs_size=[images.size()[2], images.size()[3]])
            evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, imgs_size=[images.size()[2], images.size()[3]])

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


for house in ['floor1']:
    for model_name in [globals()[f'EXP_GENERAL_DETECTOR_GIBSON_60_EPOCHS'.upper()], globals()[f'EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15'.upper()], globals()[f'EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25'.upper()], globals()[f'EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50'.upper()], globals()[f'EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75'.upper()]]:

        _, test, _, _ = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
        data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

        metrics, complete_metrics = compute_results(model_name, data_loader_test, f'Test on floor1, GD trained on gibson - Epochs GD: 60')
