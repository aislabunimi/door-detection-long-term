import cv2
import torch.optim
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.background_grid_network import BboxFilterNetworkGeometric, \
    BboxFilterNetworkGeometricLoss, BboxFilterNetworkImage, BboxFilterNetworkSuppress
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC, \
    BBOX_FILTER_NETWORK_IMAGE
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 10

dataset_creator_bboxes = DatasetCreatorBBoxes(num_bboxes=num_bboxes)

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']

train_student, validation_student, unlabelled_bbox_filter, test, labels, _ = get_final_doors_dataset_bbox_filter(folder_name='house1', train_size_student=.15)

data_loader_unlabelled = DataLoader(unlabelled_bbox_filter, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

model = YOLOv5Model(model_name=YOLOv5, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1_60_EPOCHS)

model.to('cuda')
model.model.eval()
ap_metric_classic = {0: 0, 1: 0}
complete_metric_classic = {'TP': 0, 'FP': 0, 'BFD': 0}
with torch.no_grad():
    for images, targets, converted_boxes in tqdm(data_loader_unlabelled, total=len(data_loader_unlabelled)):
        images = images.to('cuda')
        preds, train_out = model.model(images)

        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, ExampleType.TRAINING)

    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test)):
        images = images.to('cuda')
        preds, train_out = model.model(images)


        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, ExampleType.TEST)

    # Calculate metrics
    evaluator = MyEvaluator()
    evaluator_complete_metric = MyEvaluatorCompleteMetric()

    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test)):
        images = images.to('cuda')
        preds, train_out = model.model(images)
        preds = non_max_suppression(preds,
                                    0.75,
                                    0.5,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=300)

        evaluator.add_predictions_yolo(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])
        evaluator_complete_metric.add_predictions_yolo(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])
    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)
    complete_metrics = evaluator_complete_metric.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)
    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        ap_metric_classic[int(label)] = values['AP']

    total_positives = 0
    for label, values in sorted(complete_metrics.items(), key=lambda v: v[0]):
        total_positives += values['total_positives']
        complete_metric_classic['TP'] += values['TP']
        complete_metric_classic['FP'] += values['FP']
        complete_metric_classic['BFD'] += values['FPiou']
    complete_metric_classic['TP'] /= total_positives
    complete_metric_classic['FP'] /= total_positives
    complete_metric_classic['BFD'] /= total_positives

print(ap_metric_classic, complete_metric_classic)

dataset_creator_bboxes.match_bboxes_with_gt(iou_threshold_matching=0.5)
#dataset_creator_bboxes.visualize_bboxes(show_filtered=True)

train_bboxes, test_bboxes = dataset_creator_bboxes.create_datasets()

train_dataset_bboxes = DataLoader(train_bboxes, batch_size=32, collate_fn=collate_fn_bboxes, num_workers=4)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=32, collate_fn=collate_fn_bboxes, num_workers=4)

#Check the dataset
def check_bbox_dataset(dataset):
    for data in dataset:
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
        detected_bboxes = torch.transpose(detected_bboxes, 1, 2)
        images_opencv = []
        w_image, h_image = images.size()[2:][::-1]
        for image, detected_list, fixed_list, confidences_list, labels_list, ious_list in zip(images, detected_bboxes.tolist(), fixed_bboxes.tolist(), confidences.tolist(), labels_encoded.tolist(), ious.tolist()):
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image_detected = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            for (cx, cy, w, h, confidence, closed, open), suppress in zip(detected_list, confidences_list):
                print(suppress)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                image_detected = cv2.rectangle(image_detected, (x, y),
                                                       (x2, y2), (0, 0, 0) if suppress == 0 else (255, 255, 255), 2)

            images_opencv.append(image_detected)
        new_image = cv2.hconcat(images_opencv)
        cv2.imshow('show', new_image)
        cv2.waitKey()

#check_bbox_dataset(train_dataset_bboxes)
bbox_model = BboxFilterNetworkImage(fpn_channels=128, n_labels=3, model_name=BBOX_FILTER_NETWORK_IMAGE, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST)
bbox_model.to('cuda')

criterion = BboxFilterNetworkSuppress()

optimizer = optim.Adam(bbox_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion.to('cuda')
for n, p in bbox_model.named_parameters():
    if p.requires_grad:
        print(n)

logs = {'train': [], 'test': [], 'ap': {0: [], 1: []}, 'complete_metric': {'TP': [], 'FP': [], 'BFD': []}}

# compute total labels
train_total = {0:0, 1:0}
test_total = {0:0, 1:0}
for data in train_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
    for confidence_per_image in confidences.tolist():
        for confidence in confidence_per_image:
            train_total[int(confidence)] += 1

for data in test_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
    for confidence_per_image in confidences.tolist():
        for confidence in confidence_per_image:
            test_total[int(confidence)] += 1
print(test_total, train_total)
train_accuracy = {0: [], 1: []}
test_accuracy = {0: [], 1: []}

for epoch in range(60):
    #scheduler.step()
    bbox_model.train()
    criterion.train()
    optimizer.zero_grad()

    temp_losses = []
    for data in tqdm(train_dataset_bboxes, total=len(train_dataset_bboxes), desc=f'Training epoch {epoch}'):
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
        images = images.to('cuda')
        detected_bboxes = detected_bboxes.to('cuda')
        confidences = confidences.to('cuda')
        labels_encoded = labels_encoded.to('cuda')

        preds = bbox_model(images, detected_bboxes)
        #print(preds, filtered)
        loss_confidence, loss_label = criterion(preds, confidences, labels_encoded)
        final_loss = loss_confidence

        temp_losses.append(final_loss.item())
        optimizer.zero_grad()
        final_loss.backward()
        #official_loss.backward()
        optimizer.step()
    logs['train'].append(sum(temp_losses) / len(temp_losses))

    temp_losses = []

    with torch.no_grad():

        bbox_model.eval()
        criterion.eval()

        temp_accuracy = {0:0, 1:0, 2:0}
        for data in tqdm(train_dataset_bboxes, total=len(train_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
            images = images.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')

            preds = bbox_model(images, detected_bboxes)

            predicted_confidences = preds[0]

            for predicted, target in zip(predicted_confidences, confidences):

                for p_conf, gt_conf in zip(predicted.tolist(), target.tolist()):

                    if p_conf <= 0.5 and gt_conf == 0:
                        temp_accuracy[0] +=1
                    elif p_conf > 0.5 and gt_conf == 1:
                        temp_accuracy[1] +=1
        for i in range(2):
            train_accuracy[i].append(temp_accuracy[i] / train_total[i])
        temp_accuracy = {0:0, 1:0, 2:0}
        for data in tqdm(test_dataset_bboxes, total=len(test_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
            images = images.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')

            preds = bbox_model(images, detected_bboxes)
            loss_confidence, loss_label = criterion(preds, confidences, labels_encoded)
            final_loss = loss_confidence

            temp_losses.append(final_loss.item())

            predicted_confidences = preds[0]


            for predicted, target in zip(predicted_confidences, confidences):
                for p_conf, gt_conf in zip(predicted.tolist(), target.tolist()):

                    if p_conf <= 0.5 and gt_conf == 0:
                        temp_accuracy[0] +=1
                    elif p_conf > 0.5 and gt_conf == 1:
                        temp_accuracy[1] +=1

        for i in range(2):
            test_accuracy[i].append(temp_accuracy[i] / test_total[i])

    print(train_accuracy)
    fig = plt.figure()
    plt.plot([i for i in range(len(train_accuracy[0]))], train_accuracy[0], label='suppress')
    plt.plot([i for i in range(len(train_accuracy[1]))], train_accuracy[1], label='keep')

    plt.title('Train accuracy')
    plt.legend()
    plt.savefig('train.svg')

    fig = plt.figure()
    plt.plot([i for i in range(len(test_accuracy[0]))], test_accuracy[0], label='suppress')
    plt.plot([i for i in range(len(test_accuracy[1]))], test_accuracy[1], label='keep')
    plt.title('Val accuracy')
    plt.legend()
    plt.savefig('val.svg')
    print(train_accuracy)
    logs['test'].append(sum(temp_losses) / len(temp_losses))
    print(logs['train'], logs['test'])

    fig = plt.figure()
    plt.plot([i for i in range(len(logs['train']))], logs['train'], label='train_loss')
    plt.plot([i for i in range(len(logs['train']))], logs['test'], label='test_loss')
    plt.title('Losses')
    plt.legend()
    plt.savefig('losses.svg')
    bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})







