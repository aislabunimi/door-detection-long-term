import cv2
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetsCreatorBBoxes, \
    Type
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.bbox_filter_network import BboxFilterNetwork
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 15

dataset_creator_bboxes = DatasetsCreatorBBoxes(num_bboxes=num_bboxes)

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']

train_student, validation_student, unlabelled_bbox_filter, test, labels, _ = get_final_doors_dataset_bbox_filter(folder_name='house1', train_size_student=.15)

data_loader_unlabelled = DataLoader(unlabelled_bbox_filter, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

model = YOLOv5Model(model_name=YOLOv5, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1_60_EPOCHS)

model.to('cuda')
model.model.eval()
ap_metric_classic = {0: 0, 1:0}
complete_metric_classic = {'TP': 0, 'FP': 0, 'BFD': 0}
with torch.no_grad():
    for images, targets, converted_boxes in tqdm(data_loader_unlabelled, total=len(data_loader_unlabelled)):
        images = images.to('cuda')
        preds, train_out = model.model(images)
        preds = non_max_suppression(preds,
                                    0.01,
                                    0.90,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=num_bboxes)


        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, Type.TRAINING)

    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test)):
        images = images.to('cuda')
        preds, train_out = model.model(images)
        preds = non_max_suppression(preds,
                                    0.01,
                                    0.90,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=num_bboxes)


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
        ap_metric_classic[label] = values['AP']

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

dataset_creator_bboxes.filter_bboxes(iou_threshold=0.75, filter_multiple_detection=False, consider_label=False)
#dataset_creator_bboxes.visualize_bboxes(show_filtered=True)

train_bboxes, test_bboxes = dataset_creator_bboxes.create_datasets()

train_dataset_bboxes = DataLoader(train_bboxes, batch_size=4, collate_fn=collate_fn_bboxes, num_workers=4)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes, num_workers=4)

#Check the dataset
def check_bbox_dataset(dataset):
    for data in dataset:
        images, targets, converted_boxes, filtered = data
        images_opencv = []
        w_image, h_image = images.size()[2:][::-1]
        for image, target, bboxes, filter in zip(images, targets, converted_boxes, filtered):
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image_converted_bboxes = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            target_image = image_converted_bboxes.copy()
            for (cx, cy, w, h, confidence, closed, open), f in zip(bboxes.tolist(), filter.tolist()):
                print(f)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                print(x, y, x2, y2)
                if True:
                    label = 0 if closed == 1 else 1
                    image_converted_bboxes = cv2.rectangle(image_converted_bboxes, (x, y),
                                                 (x2, y2), colors[label], 2)
            for box in target['gt_bboxes']:
                x, y, w, h = box.get_absolute_bounding_box()
                print(x, y, w, h)
                label = int(box.get_class_id())
                target_image = cv2.rectangle(target_image, (int(x), int(y)),
                                                       (int(x + w), int(y + h)), colors[label], 2)

            images_opencv.append(cv2.hconcat([target_image, image_converted_bboxes]))
        new_image = cv2.vconcat(images_opencv)
        cv2.imshow('show', new_image)
        cv2.waitKey()

#check_bbox_dataset(train_dataset_bboxes)
bbox_model = BboxFilterNetwork(num_bboxes=num_bboxes, model_name=BBOX_FILTER_NETWORK, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST)
bbox_model.to('cuda')

criterion = torch.nn.MSELoss()
print(bbox_model.parameters())
optimizer = torch.optim.SGD(bbox_model.parameters(), lr=0.01)
criterion.to('cuda')
#for n, p in bbox_model.named_parameters():
#    if p.requires_grad:
#        print(n)

losses = {'train': [], 'test': []}
for epoch in range(20):
    model.train()
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
    losses['train'].append(sum(temp_losses) / len(temp_losses))
    print(losses['train'], losses['test'])






