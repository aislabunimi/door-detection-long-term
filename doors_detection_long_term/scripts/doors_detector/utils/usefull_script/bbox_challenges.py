import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_real_data


device = 'cuda'

def classify(model, dataset):
    evaluators = [MyEvaluatorCompleteMetric() for _ in range(5)]

    with torch.no_grad():
        for images, targets, converted_boxes in tqdm(dataset, total=len(dataset)):
            images = images.to(device)
            preds = model.model(images)
            preds = [apply_nms(pred, iou_thresh=0.5, confidence_threshold=0.01) for pred in preds]
            for pred in preds:
                pred['labels'] = pred['labels'] - 1
            print(targets)
            print(converted_boxes)
            print(preds)

            for t_count, target in enumerate(targets):
                for b_count, (label, (x, y, w, h)) in enumerate(zip(target['labels'], target['boxes'])):
                    x_min, y_min, x_max, y_max = np.array([x -w/2, y - h/2, x + w/2, y + h/2]) * 320
                    b = x_max - x_min
                    hh = y_max - y_min
                    ratio = min(b, hh) / max(b, hh)
                    print(ratio, int(ratio * 10))
                    evaluators[int(ratio * 5) if ratio != 1 else 4].add_predictions_faster_rcnn(targets=({'boxes': np.array([[x, y, w, h]]), 'labels': np.array([label])},), predictions=[preds[t_count]], img_size=images.size()[2:][::-1])


    metrics = [e.get_metrics(iou_threshold=0.5, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False) for e in evaluators]

    return [(m['0']['TP'] + m['1']['TP']) / (m['0']['total_positives'] + m['1']['total_positives']) for m in metrics]





for house in ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']:
    train, test, labels, _ = get_final_doors_dataset_real_data(house, 0.25)
    model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_GENERAL_DETECTOR_GIBSON_DEEP_DOORS_2_60_EPOCHS)
    model.eval()
    model.to(device)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)


    v_gd = classify(model, data_loader_test)

    model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=globals()[f'EXP_2_{house}_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15'.upper()])
    model.eval()
    model.to(device)

    v_qd = classify(model, data_loader_test)

    plt.plot([i for i in range(len(v_gd))], v_gd)
    plt.plot([i for i in range(len(v_qd))], v_qd)
    plt.show()




