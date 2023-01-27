import pandas as pd
from src.evaluators.pascal_voc_evaluator import plot_precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET, DEEP_DOORS_2_LABELLED
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.pascal_evaluator import PascalEvaluator
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.evaluators.coco_evaluator import CocoEvaluator
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything, collate_fn
from dataset_configurator import *
from doors_detection_long_term.doors_detector.models.detr_door_detector import *


device = 'cuda'
batch_size = 1

if __name__ == '__main__':
    seed_everything(0)
    houses = {
        'house1': [EXP_1_HOUSE_1, EXP_2_HOUSE_1_25, EXP_2_HOUSE_1_50, EXP_2_HOUSE_1_75],
        'house2': [EXP_1_HOUSE_2, EXP_2_HOUSE_2_25, EXP_2_HOUSE_2_50, EXP_2_HOUSE_2_75],
        'house7': [EXP_1_HOUSE_7, EXP_2_HOUSE_7_25, EXP_2_HOUSE_7_50, EXP_2_HOUSE_7_75],
        'house9': [EXP_1_HOUSE_9, EXP_2_HOUSE_9_25, EXP_2_HOUSE_9_50, EXP_2_HOUSE_9_75],
        'house10': [EXP_1_HOUSE_10, EXP_2_HOUSE_10_25, EXP_2_HOUSE_10_50, EXP_2_HOUSE_10_75],
        'house13': [EXP_1_HOUSE_13, EXP_2_HOUSE_13_25, EXP_2_HOUSE_13_50, EXP_2_HOUSE_13_75],
        'house15': [EXP_1_HOUSE_15, EXP_2_HOUSE_15_25, EXP_2_HOUSE_15_50, EXP_2_HOUSE_15_75],
        'house20': [EXP_1_HOUSE_20, EXP_2_HOUSE_20_25, EXP_2_HOUSE_20_50, EXP_2_HOUSE_20_75],
        'house21': [EXP_1_HOUSE_21, EXP_2_HOUSE_21_25, EXP_2_HOUSE_21_50, EXP_2_HOUSE_21_75],
        'house22': [EXP_1_HOUSE_22, EXP_2_HOUSE_22_25, EXP_2_HOUSE_22_50, EXP_2_HOUSE_22_75],
    }
    house = 'house22'
    door_no_door_task = True
    train, test, labels, COLORS = get_final_doors_dataset(experiment=2, folder_name=house, train_size=0.25, use_negatives=True)
    #train, test, labels, COLORS = get_deep_doors_2_labelled_sets()

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    if door_no_door_task:
        metrics_table = pd.DataFrame(data={
            'Env': [house for _ in range(8)],
            'Exp':  ['1', '1', '2a', '2a', '2b', '2b', '2c', '2c'],
            'Label': ['-1', '0', '-1', '0', '-1', '0', '-1', '0'],
            'AP': [0.0 for _ in range(8)],
            'Positives': [0 for _ in range(8)],
            'TP': [0 for _ in range(8)],
            'FP': [0 for _ in range(8)]
        })
    else:
        metrics_table = pd.DataFrame(data={
            'Env': [house for _ in range(12)],
            'Exp':  ['1', '1', '1', '2a', '2a', '2a', '2b', '2b', '2b', '2c', '2c', '2c'],
            'Label': ['-1', '0', '1', '-1', '0', '1', '-1', '0', '1', '-1', '0', '1'],
            'AP': [0.0 for _ in range(12)],
            'Positives': [0 for _ in range(12)],
            'TP': [0 for _ in range(12)],
            'FP': [0 for _ in range(12)]
        })

    for index, exp in enumerate(houses[house]):
        model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=exp)

        model.eval()
        model.to(device)

        data_loader_test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, drop_last=False, num_workers=4)
        evaluator = MyEvaluator()

        for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
            images = images.to(device)
            outputs = model(images)
            evaluator.add_predictions(targets=targets, predictions=outputs)

        metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.5, door_no_door_task=door_no_door_task, plot_curves=True, colors=COLORS)
        mAP = 0
        print('Results per bounding box:')
        for i, (label, values) in enumerate(sorted(metrics['per_bbox'].items(), key=lambda v: v[0])):
            m = 2 if door_no_door_task else 3
            mAP += values['AP']
            print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
            print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
            metrics_table['AP'][index * m + i] = values['AP']
            metrics_table['Positives'][index * m + i] = values['total_positives']
            metrics_table['TP'][index * m + i] = values['TP']
            metrics_table['FP'][index * m + i] = values['FP']
        print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')

        mAP = 0
        print('Results per image')
        for label, values in sorted(metrics['per_image'].items(), key=lambda v: v[0]):
            print(f'\tLabel {label} -> Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}, FN = {values["FN"]}')
            print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    final = '_door_no_door.xlsx' if door_no_door_task else '.xlsx'
    with pd.ExcelWriter('./../results/'+house + final) as writer:
        if not metrics_table.index.name:
            metrics_table.index.name = 'Index'
        metrics_table.to_excel(writer, sheet_name='s')