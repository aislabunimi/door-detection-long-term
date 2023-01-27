import time

import pandas as pd
from models.detr import SetCriterion
from models.matcher import HungarianMatcher
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_door_fine_tune_model_output import \
    DatasetCreatorFineTuneModelOutput
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_experiment_k import DatasetsCreatorExperimentK
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_experiment_k_ordered import \
    DatasetsCreatorExperimentKOrdered
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.experiment_k.criterion import CriterionType, Criterion, CriterionSorting
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.plot import plot_losses
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything, collate_fn
from dataset_configurator import get_final_doors_dataset_door_no_door_task
import torch.nn.functional as F


params = {
    'seed': 0,
    'batch_size': 1,
    'epochs': 10,
    'lr': 1e-5,
    'weight_decay': 1e-4,
    'lr_drop': 20,
    'lr_backbone': 1e-6,
    # Criterion
    'bbox_loss_coef': 5,
    'giou_loss_coef': 2,
    'eos_coef': 0.1,
    # Matcher
    'set_cost_class': 1,
    'set_cost_bbox': 5,
    'set_cost_giou': 2,
}

seed_everything(params['seed'])

door_no_door_task = False

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

percentage = 0.2
threshold_bboxes = 0.5
criterion_type = CriterionType.MIN
image_criterion = Criterion(criterion_type=criterion_type)

for house in houses.keys():
    if door_no_door_task:
        metrics_table_first = pd.DataFrame(data={
            'Env': [house for _ in range(2)],
            'Percentage':  ['0', '0'],
            'Label': ['-1', '0'],
            'AP': [0.0 for _ in range(2)],
            'Positives': [0 for _ in range(2)],
            'TP': [0 for _ in range(2)],
            'FP': [0 for _ in range(2)]
        })
    else:
        metrics_table_first = pd.DataFrame(data={
            'Env': [house for _ in range(3)],
            'Percentage':  ['0', '0', '0'],
            'Label': ['-1', '0', '1'],
            'AP': [0.0 for _ in range(3)],
            'Positives': [0 for _ in range(3)],
            'TP': [0 for _ in range(3)],
            'FP': [0 for _ in range(3)]
        })

    train, test, labels_set, COLORS = get_final_doors_dataset_door_no_door_task(folder_name=house, train_size=0.75, test_size=0.25)

    K = len(train)

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels_set.keys()), pretrained=True,
                                     dataset_name=FINAL_DOORS_DATASET, description=houses[house][0])

    # Select the images that respect the criterion
    data_loader_classify = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)

    dataset_model_output = DatasetsCreatorExperimentKOrdered(
        dataset_path='/home/michele/myfiles/final_doors_dataset',
        folder_name=house,
        test_dataframe=test.get_dataframe()
    )

    for i, (images, targets) in tqdm(enumerate(data_loader_classify), total=len(data_loader_classify), desc='Collect model output'):

        outputs = model(images)
        pred_logits, pred_boxes_images = outputs['pred_logits'], outputs['pred_boxes']
        prob = F.softmax(pred_logits, -1)

        scores_images, labels_images = prob[..., :-1].max(-1)

        for i, (scores, labels, bboxes) in enumerate(zip(scores_images, labels_images, pred_boxes_images)):
            keep = scores >= threshold_bboxes
            scores = scores[keep]
            labels = labels[keep]
            bboxes = bboxes[keep]
            score_image = image_criterion.aggregate_score_image(scores)

            dataset_model_output.add_train_sample(absolute_count=targets[i]['absolute_count'], score_image=score_image)

    # Test
    data_loader_test = DataLoader(test, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)
    evaluator = MyEvaluator()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
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
        metrics_table_first['AP'][i] = values['AP']
        metrics_table_first['Positives'][i] = values['total_positives']
        metrics_table_first['TP'][i] = values['TP']
        metrics_table_first['FP'][i] = values['FP']
    print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')

    for sorting in [CriterionSorting.DESCENDING, CriterionSorting.GROWING]:
        metrics_table = metrics_table_first
        # Fine tune and test for different percentage of k
        for percentage in [0.2 * i for i in range(1, 6)]:
            print(percentage)

            model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels_set.keys()), pretrained=True,
                                    dataset_name=FINAL_DOORS_DATASET, description=houses[house][0])

            train, test = dataset_model_output.create_datasets(number_of_samples=int(round(percentage * K)), sorting=sorting, random_state=42)

            data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)
            data_loader_test_model = DataLoader(test, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)

            # Fine tune
            losses = ['labels', 'boxes', 'cardinality']
            weight_dict = {'loss_ce': 1, 'loss_bbox': params['bbox_loss_coef'], 'loss_giou': params['giou_loss_coef']}
            matcher = HungarianMatcher(cost_class=params['set_cost_class'], cost_bbox=params['set_cost_bbox'], cost_giou=params['set_cost_giou'])
            criterion = SetCriterion(len(labels_set.keys()), matcher=matcher, weight_dict=weight_dict,
                                     eos_coef=params['eos_coef'], losses=losses)

            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": params['lr_backbone'],
                },
            ]

            optimizer = torch.optim.AdamW(param_dicts, lr=params['lr'], weight_decay=params['weight_decay'])
            logs = {'train': [], 'test': [], 'time': []}
            device = 'cuda'
            model.to(device)
            criterion.to(device)
            print_logs_every = 10
            start_time = time.time()

            for epoch in range(params['epochs']):

                temp_logs = {'train': [], 'test': []}
                accumulate_losses = {}

                model.train()
                criterion.train()

                for i, training_data in enumerate(data_loader_train):
                    # Data is a tuple where
                    #   - data[0]: a tensor containing all images shape = [batch_size, channels, img_height, img_width]
                    #   - data[1]: a tuple of dictionaries containing the images' targets

                    images, targets = training_data
                    images = images.to(device)

                    # Move targets to device
                    targets = [{k: v.to(device) for k, v in target.items() if k != 'folder_name' and k != 'absolute_count'} for target in targets]
                    outputs = model(images)

                    # Compute losses
                    losses_dict = criterion(outputs, targets)
                    weight_dict = criterion.weight_dict

                    # Losses are weighted using parameters contained in a dictionary
                    losses = sum(losses_dict[k] * weight_dict[k] for k in losses_dict.keys() if k in weight_dict)

                    scaled_losses_dict = {k: v * weight_dict[k] for k, v in losses_dict.items() if k in weight_dict}
                    scaled_losses_dict['loss'] = sum(scaled_losses_dict.values())

                    # Back propagate the losses
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    accumulate_losses = {k: scaled_losses_dict[k] if k not in accumulate_losses else sum([accumulate_losses[k], scaled_losses_dict[k]]) for k, _ in scaled_losses_dict.items()}

                    if i % print_logs_every == print_logs_every - 1:
                        accumulate_losses = {k: v.item() / print_logs_every for k, v in accumulate_losses.items()}
                        print(f'Train epoch [{epoch}] -> [{i}/{len(data_loader_train)}]: ' + ', '.join([f'{k}: {v}' for k, v in accumulate_losses.items()]))
                        temp_logs['train'].append(accumulate_losses)
                        accumulate_losses = {}

                epoch_total = {}
                for d in temp_logs['train']:
                    for k in d:
                        epoch_total[k] = epoch_total.get(k, 0) + d[k]

                logs['train'].append({k: v / len(temp_logs['train']) for k, v in epoch_total.items()})

                print(f'----> EPOCH SUMMARY TRAIN [{epoch}] -> [{i}/{len(data_loader_train)}]: ' + ', '.join([f'{k}: {v}' for k, v in logs['train'][epoch].items()]))

                with torch.no_grad():
                    model.eval()
                    criterion.eval()

                    accumulate_losses = {}
                    for i, test_data in enumerate(data_loader_test_model):
                        images, targets = test_data
                        images = images.to(device)

                        # Move targets to device
                        targets = [{k: v.to(device) for k, v in target.items() if k != 'folder_name' and k != 'absolute_count'} for target in targets]

                        outputs = model(images)

                        # Compute losses
                        losses_dict = criterion(outputs, targets)
                        weight_dict = criterion.weight_dict

                        # Losses are weighted using parameters contained in a dictionary
                        losses = sum(losses_dict[k] * weight_dict[k] for k in losses_dict.keys() if k in weight_dict)

                        scaled_losses_dict = {k: v * weight_dict[k] for k, v in losses_dict.items() if k in weight_dict}
                        scaled_losses_dict['loss'] = sum(scaled_losses_dict.values())
                        accumulate_losses = {k: scaled_losses_dict[k] if k not in accumulate_losses else sum([accumulate_losses[k], scaled_losses_dict[k]]) for k, _ in scaled_losses_dict.items()}

                        if i % print_logs_every == print_logs_every - 1:
                            accumulate_losses = {k: v.item() / print_logs_every for k, v in accumulate_losses.items()}
                            print(f'Test epoch [{epoch}] -> [{i}/{len(data_loader_test)}]: ' + ', '.join([f'{k}: {v}' for k, v in accumulate_losses.items()]))
                            temp_logs['test'].append(accumulate_losses)
                            accumulate_losses = {}

                epoch_total = {}
                for d in temp_logs['test']:
                    for k in d:
                        epoch_total[k] = epoch_total.get(k, 0) + d[k]

                logs['test'].append({k: v / len(temp_logs['test']) for k, v in epoch_total.items()})
                logs['time'].append(time.time() - start_time)

                print(f'----> EPOCH SUMMARY TEST [{epoch}] -> [{i}/{len(data_loader_test_model)}]: ' + ', '.join([f'{k}: {v}' for k, v in logs['test'][epoch].items()]))

                #lr_scheduler.step()

                #plot_losses(logs)

            # Test
            if door_no_door_task:
                new_metrics_table = pd.DataFrame(data={
                    'Env': [house for _ in range(2)],
                    'Percentage':  [str(percentage), str(percentage)],
                    'Label': ['-1', '0'],
                    'AP': [0.0 for _ in range(2)],
                    'Positives': [0 for _ in range(2)],
                    'TP': [0 for _ in range(2)],
                    'FP': [0 for _ in range(2)]
                })
            else:
                new_metrics_table = pd.DataFrame(data={
                    'Env': [house for _ in range(3)],
                    'Percentage':  [str(percentage), str(percentage), str(percentage)],
                    'Label': ['-1', '0', '1'],
                    'AP': [0.0 for _ in range(3)],
                    'Positives': [0 for _ in range(3)],
                    'TP': [0 for _ in range(3)],
                    'FP': [0 for _ in range(3)]
                })

            model.to('cpu')
            model.eval()
            evaluator = MyEvaluator()

            for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
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
                new_metrics_table['AP'][i] = values['AP']
                new_metrics_table['Positives'][i] = values['total_positives']
                new_metrics_table['TP'][i] = values['TP']
                new_metrics_table['FP'][i] = values['FP']
            print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')
            print(new_metrics_table)
            metrics_table = metrics_table.append(new_metrics_table, ignore_index=True)
            #percentage *= 2

        final = '_experimentk_ordered_' + str(criterion_type) + '_' + str(sorting)
        final += '_door_no_door.xlsx' if door_no_door_task else '.xlsx'
        with pd.ExcelWriter('./../results/'+ house + final) as writer:
            if not metrics_table.index.name:
                metrics_table.index.name = 'Index'
            metrics_table.to_excel(writer, sheet_name='s')