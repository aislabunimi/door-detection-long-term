import os

import cv2
import numpy as np
import torch
import torchvision

from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import xywh2xyxy

#Check the dataset
def check_bbox_dataset(dataset, confidence_threshold):
    colors = {0: (0, 0, 255), 1: (0, 255, 0)}
    for i, data in enumerate(dataset):
        if i < 2900:
            continue
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
        detected_bboxes = torch.transpose(detected_bboxes, 1, 2)
        images_opencv = []
        w_image, h_image = images.size()[2:][::-1]
        for image, detected_list, fixed_list, confidences_list, labels_list, ious_list, target_boxes_list in zip(images, detected_bboxes.tolist(), fixed_bboxes.tolist(), confidences.tolist(), labels_encoded.tolist(), ious.tolist(), target_boxes):
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image_detected = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            image_detected_high_conf = image_detected.copy()
            image_correct_label = image_detected.copy()
            image_target = image_detected.copy()
            for (cx, cy, w, h, confidence, closed, open), (back, closed, open) in zip(detected_list, labels_list):
                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                print(x, y, x2, y2)
                label = 0 if closed == 1 else 1
                image_detected = cv2.rectangle(image_detected, (x, y),
                                               (x2, y2), colors[label], 2)
                if confidence >= confidence_threshold:
                    image_detected_high_conf = cv2.rectangle(image_detected_high_conf, (x, y),
                                                             (x2, y2), colors[label], 2)

                if back == 1:
                    color = (0,0,0)
                elif closed == 1:
                    color= ( 0, 0, 255)
                else:
                    color =( 0,255, 0)

                image_correct_label = cv2.rectangle(image_correct_label, (x, y),
                                                    (x2, y2), color, 2)

            for cx, cy, w, h, label in target_boxes_list:
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                print(x, y, x2, y2)
                label = int(label)
                image_target = cv2.rectangle(image_target, (x, y),
                                             (x2, y2), colors[label], 2)

            images_opencv.append(cv2.hconcat([image_target, image_detected, image_detected_high_conf, image_correct_label]))
        new_image = cv2.vconcat(images_opencv)
        cv2.imshow('show', new_image)
        cv2.waitKey()


def plot_results(epoch, count, env, images, bboxes, preds, targets, confidence_threshold):
    if not os.path.exists('/home/antonazzi/myfiles/bbox_filtering/'+str(epoch)):
        os.makedirs('/home/antonazzi/myfiles/bbox_filtering/'+str(epoch))
    if not os.path.exists('/home/antonazzi/myfiles/bbox_filtering/'+str(epoch) + f'/{env}'):
        os.makedirs('/home/antonazzi/myfiles/bbox_filtering/'+str(epoch) + f'/{env}')
    colors = {0: (0, 0, 1), 1: (0, 1, 0)}

    for image, bboxes_image, confidences, labels, target in zip(images, bboxes, preds[0], preds[1], targets):
        bboxes_image = bboxes_image.transpose(0, 1)
        w_image, h_image = image.size()[1:][::-1]
        image = image.to('cpu')
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image_detected = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
        target_image = image_detected.copy()
        for (cx, cy, w, h), confidence, classes in zip(bboxes_image[:, :4].tolist(), confidences.tolist(), labels.tolist()):
            #print(cx, cy, w, h)
            x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
            x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
            label = classes.index(max(classes))
            if label == 0 or confidence < confidence_threshold:
                continue
            label -= 1
            image_detected = cv2.rectangle(image_detected, (x, y),
                                               (x2, y2), colors[label], 2)

        for cx, cy, w, h, label in target.tolist():
            x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
            x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
            label = int(label)
            target_image = cv2.rectangle(target_image, (x, y),
                                             (x2, y2), colors[label], 2)

        image = cv2.hconcat([target_image, image_detected])
        cv2.imwrite('/home/antonazzi/myfiles/bbox_filtering/'+str(epoch) + f'/{env}'+f"/{count}.png", (image*255).astype(np.uint8))


def bounding_box_filtering_yolo(predictions, max_detections, iou_threshold=0.5, confidence_threshold=0.01, apply_nms: bool = False):
    bboxes = []
    for image_prediction in predictions:
        coords = xywh2xyxy(image_prediction[:, :4])
        conf = image_prediction[:, 5:] * image_prediction[:, 4:5]
        conf, labels = conf.max(1, keepdim=True)
        conf = torch.squeeze(conf)
        labels = torch.squeeze(labels)
        if apply_nms:
            i = torchvision.ops.nms(coords, conf, iou_threshold=iou_threshold)
            i = i[:max_detections]

        else:
            i = torch.argsort(conf, descending=True)
            i = i[:max_detections]
        coords = coords[i]
        conf = conf[i]
        labels = labels[i]

        keep_conf = conf > confidence_threshold
        conf = conf[keep_conf]
        labels = labels[keep_conf]
        coords = coords[keep_conf, :]
        bboxes.append(torch.cat([coords, conf.unsqueeze(1), labels.unsqueeze(1)], dim=1))
    return bboxes

def bounding_box_filtering_after_network(detected_bboxes, preds, image_size, iou_threshold=0.5):
    bboxes = []
    new_confidences = []
    new_labels = []
    for bboxes_image, original_bboxes_image, confidences, labels in zip(detected_bboxes.transpose(1, 2).clone().detach(), detected_bboxes.transpose(1, 2), preds[0], preds[1]):
        bboxes_image[:, :4] *= torch.tensor([image_size + image_size], device=bboxes_image.device)
        coords = xywh2xyxy(torch.round(bboxes_image[:, :4]))
        i = torchvision.ops.nms(coords, confidences, iou_threshold=iou_threshold)
        bboxes.append(original_bboxes_image[i].transpose(0, 1))
        new_confidences.append(confidences[i])
        new_labels.append(labels[i])

    return bboxes, (new_confidences, new_labels)