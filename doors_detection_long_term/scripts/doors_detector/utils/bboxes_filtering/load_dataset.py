import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetsCreatorBBoxes
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_bboxes

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 20

dataset_creator_bboxes = DatasetsCreatorBBoxes(max_bboxes=num_bboxes)
dataset_creator_bboxes.load_dataset(folder_name='yolov5_simulation_dataset')

dataset_creator_bboxes.select_n_bounding_boxes(num_bboxes=20)
dataset_creator_bboxes.match_bboxes_with_gt(iou_threshold_matching=0.75)

train_bboxes, test_bboxes = dataset_creator_bboxes.create_datasets(shuffle_boxes=False)

train_dataset_bboxes = DataLoader(train_bboxes, batch_size=2, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4, shuffle=True)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=2, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4)

#Check the dataset
def check_bbox_dataset(dataset):
    for data in dataset:
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
        detected_bboxes = torch.transpose(detected_bboxes, 1, 2)
        images_opencv = []
        w_image, h_image = images.size()[2:][::-1]
        for image, detected_list, fixed_list, confidences_list, labels_list, ious_list, target_boxes_list in zip(images, detected_bboxes.tolist(), fixed_bboxes.tolist(), confidences.tolist(), labels_encoded.tolist(), ious.tolist(), target_boxes):
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image_detected = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            image_detected_correct_label = image_detected.copy()
            image_matched = image_detected.copy()
            for (cx, cy, w, h, confidence, closed, open), (cx_f, cy_f, w_f, h_f), (back, closed, open) in zip(detected_list, fixed_list, labels_list):
                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                print(x, y, x2, y2)
                label = 0 if closed == 1 else 1
                image_detected = cv2.rectangle(image_detected, (x, y),
                                               (x2, y2), colors[label], 2)

                c = (0, 0, 0)
                if closed:
                    c = (0,0,255)
                elif open == 1:
                    c = 0,255, 0
                image_detected_correct_label = cv2.rectangle(image_detected_correct_label, (x, y),
                                                             (x2, y2), c, 2)

                if not(cx == cx_f and cy == cy_f and w == w_f and h == h_f):
                    image_matched = cv2.rectangle(image_matched, (x, y),
                                                  (x2, y2), colors[label], 2)
                else:
                    image_matched = cv2.rectangle(image_matched, (x, y),
                                                  (x2, y2),(0,0,0), 2)

            images_opencv.append(cv2.hconcat([image_detected, image_detected_correct_label, image_matched]))
        new_image = cv2.vconcat(images_opencv)
        cv2.imshow('show', new_image)
        cv2.waitKey()

check_bbox_dataset(test_dataset_bboxes)