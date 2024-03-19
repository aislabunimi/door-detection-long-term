import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetLoaderBBoxes import DatasetLoaderBBoxes
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.background_grid_network import *
from doors_detection_long_term.doors_detector.models.bbox_filter_network_geometric import *
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_bboxes
colors = {0: (0, 0, 255), 1: (0, 255, 0)}
torch.autograd.set_detect_anomaly(True)

house = 'chemistry_floor0'
save_path = f"/home/antonazzi/myfiles/video/{house}"
save_tasknet = save_path + '/tasknet'
if not os.path.exists(save_tasknet):
    os.makedirs(save_tasknet)


quantity = 0.75
num_bboxes = 30
iou_threshold_matching = 0.5
confidence_threshold_original = 0.75

save_filternet = save_path + f'/r2snet_{quantity}_{num_bboxes}'
if not os.path.exists(save_filternet):
    os.makedirs(save_filternet)

grid_dim = [(2**i, 2**i) for i in range(3, 6)][::-1]
device = 'cuda'
filter_description = globals()[f'IMAGE_NETWORK_GEOMETRIC_BACKGROUND_GIBSON_DD2_FINE_TUNE_{house}_{int(quantity*100)}_bbox_{num_bboxes}'.upper()]
print(filter_description)

dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name=f'faster_rcnn_general_detector_gibson_dd2_{house}_bag')
train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=False, shuffle_boxes=False, shuffle_images=False)

test_dataset_bboxes = DataLoader(train_bboxes, batch_size=16, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)

bbox_model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=True, grid_network_pretrained=False, dataset_name=FINAL_DOORS_DATASET,
                                                  description=filter_description, description_background=IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL)

bbox_model.to(device)

with torch.no_grad():
    bbox_model.eval()
    i = 0
    evaluator_complete_metric = MyEvaluatorCompleteMetric()
    evaluator_ap = MyEvaluator()

    evaluator_complete_metric_tasknet = MyEvaluatorCompleteMetric()
    evaluator_ap_tasknet = MyEvaluator()
    for data in tqdm(test_dataset_bboxes, ):
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
        #print(detected_bboxes[0])
        # Inference
        images = images.to(device)
        for k, v in image_grids.items():
            image_grids[k] = v.to(device)

        for k, v in detected_boxes_grid.items():
            detected_boxes_grid[k] = v.to(device)

        detected_bboxes_cuda = detected_bboxes.to(device)

        preds = bbox_model(images, detected_bboxes_cuda, detected_boxes_grid)
        #print(preds[0])


        new_labels, new_labels_indexes = torch.max(preds[0].to('cpu'), dim=2, keepdim=False)
        detected_bboxes_predicted = detected_bboxes_cuda.transpose(1, 2).to('cpu')

        # Modify confidences according to the model output

        new_confidences = preds[2]
        _, new_confidences_indexes = torch.max(new_confidences, dim=2)
        new_confidences_indexes = new_confidences_indexes
        new_confidences_indexes[new_confidences_indexes < 0] = 0
        new_confidences_indexes[new_confidences_indexes > 9] = 9
        new_confidences_indexes = new_confidences_indexes * 0.1

        detected_bboxes_predicted[:, :, 4] = new_confidences_indexes

        # Remove bboxes with background network
        new_labels_indexes[torch.max(preds[1], dim=2)[1] == 0] = 0

        # Filtering bboxes according to new labels
        detected_bboxes_predicted = torch.unbind(detected_bboxes_predicted, 0)
        detected_bboxes_predicted = [b[i != 0, :] for b, i in zip(detected_bboxes_predicted, new_labels_indexes)]

        detected_bboxes_predicted = [torch.cat([b[:, :5], p[i != 0][:, 1:]], dim=1) for b, p, i in zip(detected_bboxes_predicted, preds[0].to('cpu'), new_labels_indexes)]
        # Delete bboxes according to the background network
        detected_bboxes_predicted = bbox_filtering_nms(detected_bboxes_predicted, confidence_threshold=0, iou_threshold=0.5, img_size=images.size()[::-1][:2])
        evaluator_complete_metric.add_predictions_bboxes_filtering(bboxes=detected_bboxes_predicted, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
        evaluator_ap.add_predictions_bboxes_filtering(bboxes=detected_bboxes_predicted, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])


        detected_bboxes = detected_bboxes.transpose(1, 2)
        detected_bboxes_tasknet = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold_original, iou_threshold=0.5, img_size=images.size()[::-1][:2])
        evaluator_ap_tasknet.add_predictions_bboxes_filtering(detected_bboxes_tasknet, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
        evaluator_complete_metric_tasknet.add_predictions_bboxes_filtering(detected_bboxes_tasknet, target_bboxes=target_boxes, img_size=images.size()[::-1][:2])
        for image, detected_bboxes_tasknet_image, target_boxes_image, detected_bboxes_image, labels_encoded_image, detected_bboxes_predicted_image in zip(images, detected_bboxes_tasknet, target_boxes, detected_bboxes, labels_encoded, detected_bboxes_predicted):
            w_image, h_image = image.size()[1:][::-1]
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image = (np.transpose(np.array(image), (1, 2, 0))[:280, :]* 255).astype(np.uint8)

            image[0:40, :] = [255, 255, 255]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_tasknet = np.copy(image)

            list_coords = []
            for (cx, cy, w, h, c, closed, open) in detected_bboxes_tasknet_image.tolist():

                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                list_coords.append([max(1, x), max(41, y),min(x2-x, w_image-2),min(239, y2- y), c, closed, open])
            for x, y, w, h, c, closed, open in list_coords:

                image_tasknet = cv2.rectangle(image_tasknet, (x, y), (x +w, y + h), (0, 0, 255) if closed else (0, 255, 0), 2)
            cv2.imwrite(save_tasknet + f'/{i}.jpeg', image_tasknet[40:, :])

            for (cx, cy, w, h, c, closed, open) in detected_bboxes_predicted_image.tolist():
                labels = [round(closed, 8), round(open, 8)]
                label = labels.index(max(labels))

                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)

                if label == 0:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                image = cv2.rectangle(image, (max(1,x), max(41, y)), (min(x2, w_image-2),min(279, y2)), color, 2)
            cv2.imwrite(save_filternet + f'/{i}.jpeg', image[40:, :])
            i+=1

        metrics = evaluator_complete_metric.get_metrics(confidence_threshold=0.38, iou_threshold=0.5)
        metrics_ap = evaluator_ap.get_metrics(confidence_threshold=0.38, iou_threshold=0.5)

        metrics_tasknet = evaluator_complete_metric_tasknet.get_metrics(confidence_threshold=0.75, iou_threshold=0.5)
        metrics_ap_tasknet = evaluator_ap_tasknet.get_metrics(confidence_threshold=0.75, iou_threshold=0.5)

        performance = {'tasknet': {}, 'filternet': {}}
        performance_ap = {'tasknet': {}, 'filternet': {}}
        for (label_t, values_t), (label, values) in zip(metrics_tasknet.items(), metrics.items()):
            for k, v in values_t.items():
                if k not in performance['tasknet']:
                    performance['tasknet'][k] = v
                else:
                    performance['tasknet'][k] += v
            for k, v in values.items():
                if k not in performance['filternet']:
                    performance['filternet'][k] = v
                else:
                    performance['filternet'][k] += v

        for (label_t, v_t), (label, v) in zip(metrics_ap_tasknet['per_bbox'].items(), metrics_ap['per_bbox'].items()):
            performance_ap['tasknet'][label_t] = v_t['AP']
            performance_ap['filternet'][label] = v['AP']

        print(performance, performance_ap)


