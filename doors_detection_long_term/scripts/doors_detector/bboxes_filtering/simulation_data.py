import cv2
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetsCreatorBBoxes, \
    Type
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.bbox_filter_network import BboxFilterNetworkGeometric, BboxFilterNetworkGeometricLoss
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 50

dataset_creator_bboxes = DatasetsCreatorBBoxes(num_bboxes=num_bboxes)

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

        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, Type.TRAINING)

    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test)):
        images = images.to('cuda')
        preds, train_out = model.model(images)


        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, Type.TEST)

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

print(train_bboxes[0])

train_dataset_bboxes = DataLoader(train_bboxes, batch_size=4, collate_fn=collate_fn_bboxes, num_workers=4)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes, num_workers=4)

#Check the dataset
def check_bbox_dataset(dataset):
    for data in dataset:
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
        images_opencv = []
        w_image, h_image = images.size()[2:][::-1]
        for image, detected_list, fixed_list, confidences_list, labels_list, ious_list in zip(images, detected_bboxes.tolist(), fixed_bboxes.tolist(), confidences.tolist(), labels_encoded.tolist(), ious.tolist()):
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image_detected = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            image_detected_high_conf = image_detected.copy()
            image_matched = image_detected.copy()
            for (cx, cy, w, h, confidence, closed, open), (cx_f, cy_f, w_f, h_f) in zip(detected_list, fixed_list):
                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                print(x, y, x2, y2)
                label = 0 if closed == 1 else 1
                image_detected = cv2.rectangle(image_detected, (x, y),
                                                       (x2, y2), colors[label], 2)
                if confidence >= 0.75:
                    image_detected_high_conf = cv2.rectangle(image_detected_high_conf, (x, y),
                                                   (x2, y2), colors[label], 2)

                if not(cx == cx_f and cy == cy_f and w == w_f and h == h_f):
                    image_matched = cv2.rectangle(image_matched, (x, y),
                                                   (x2, y2), colors[label], 2)
                else:
                    image_matched = cv2.rectangle(image_matched, (x, y),
                                                  (x2, y2),(0,0,0), 2)

            images_opencv.append(cv2.hconcat([image_detected, image_detected_high_conf, image_matched]))
        new_image = cv2.vconcat(images_opencv)
        cv2.imshow('show', new_image)
        cv2.waitKey()

#check_bbox_dataset(train_dataset_bboxes)
bbox_model = BboxFilterNetwork(num_bboxes=num_bboxes, model_name=BBOX_FILTER_NETWORK, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST)
bbox_model.to('cuda')

criterion = BboxFilterNetworkLoss(reduction_image='sum', reduction_global='mean')
print(bbox_model.parameters())
optimizer = torch.optim.SGD(bbox_model.parameters(), lr=0.01)
criterion.to('cuda')
#for n, p in bbox_model.named_parameters():
#    if p.requires_grad:
#        print(n)

logs = {'train': [], 'test': [], 'ap': {0: [], 1: []}, 'complete_metric': {'TP': [], 'FP': [], 'BFD': []}}

for epoch in range(20):
    bbox_model.train()
    criterion.train()
    optimizer.zero_grad()

    temp_losses = []
    for data in tqdm(train_dataset_bboxes, total=len(train_dataset_bboxes), desc=f'Training epoch {epoch}'):
        images, targets, converted_boxes, filtered = data
        images = images.to('cuda')
        converted_boxes = converted_boxes.to('cuda')
        filtered = filtered.to('cuda')

        preds = bbox_model(images, converted_boxes)
        #print(preds, filtered)
        loss = criterion(preds, filtered)
        temp_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    logs['train'].append(sum(temp_losses) / len(temp_losses))

    temp_losses = []


    with torch.no_grad():
        bbox_model.eval()
        criterion.eval()
        for data in tqdm(test_dataset_bboxes, total=len(test_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, targets, converted_boxes, filtered = data
            images = images.to('cuda')
            converted_boxes = converted_boxes.to('cuda')
            filtered = filtered.to('cuda')

            preds = bbox_model(images, converted_boxes)
            #print(preds, filtered)
            loss = criterion(preds, filtered)
            temp_losses.append(loss.item())

    logs['test'].append(sum(temp_losses) / len(temp_losses))
    print(logs['train'], logs['test'])
    bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})







