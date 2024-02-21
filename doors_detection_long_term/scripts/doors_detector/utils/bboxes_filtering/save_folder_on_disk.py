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

save_path = "/home/antonazzi/myfiles/save_bbox_filtering"

house = 'floor1'

quantity = 0.25
num_bboxes = 100
iou_threshold_matching = 0.5
confidence_threshold_original = 0.75
grid_dim = [(2**i, 2**i) for i in range(3, 6)][::-1]
device = 'cuda'
filter_description = globals()[f'IMAGE_NETWORK_GEOMETRIC_BACKGROUND_GIBSON_DD2_FINE_TUNE_{house.replace("_evening", "")}_{int(quantity*100)}_bbox_{num_bboxes}'.upper()]
print(filter_description)

dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name=f'faster_rcnn_general_detector_gibson_dd2_{house}_{quantity}')
train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)

test_dataset_bboxes = DataLoader(test_bboxes, batch_size=16, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)

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
            image = np.transpose(np.array(image), (1, 2, 0))[:280,:]
            image[0:40, :] = [1, 1, 1]



            fig, ((gt_image, image_original, tasknet_prediction_unfiltered,),
                  ( correct_predicitions_unfiltered, wrong_predicitions_unfiltered, all_predicitions_unfiltered),
                  (predicted_filternet, predicted_filternet_nms, _)) = plt.subplots(3, 3)
            image_original.imshow((image*255).astype(np.uint8))
            image_original.axis('off')

            # GT
            gt_image.imshow((image*255).astype(np.uint8))
            gt_image.axis('off')

            # Tasknet predicition
            tasknet_prediction_unfiltered.imshow((image*255).astype(np.uint8))
            tasknet_prediction_unfiltered.axis('off')

            correct_predicitions_unfiltered.imshow((image*255).astype(np.uint8))
            correct_predicitions_unfiltered.axis('off')

            # Wrong prediction unfiltered
            wrong_predicitions_unfiltered.imshow((image*255).astype(np.uint8))
            wrong_predicitions_unfiltered.axis('off')

            # Wrong prediction unfiltered
            all_predicitions_unfiltered.imshow((image*255).astype(np.uint8))
            all_predicitions_unfiltered.axis('off')

            # Wrong prediction unfiltered
            predicted_filternet.imshow((image*255).astype(np.uint8))
            predicted_filternet.axis('off')

            # Filternet output

            for (cx, cy, w, h, label) in target_boxes_image:
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)

                gt_image.add_patch(Rectangle((max(x, 2), max(42, y)),min(x2-x, w_image-2),min(237, y2- y),
                                                   edgecolor=(1, 0, 0) if label == 0 else (0, 1, 0),
                                                   facecolor='none',
                                                   lw=3))
            # Detected by the plain network (without filternet)

            list_coords = []
            for (cx, cy, w, h, c, closed, open) in detected_bboxes_tasknet_image.tolist():

                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                list_coords.append([max(2, x), max(42, y),min(x2-x, w_image-2),min(237, y2- y), c, closed, open])
            for x, y, w, h, c, closed, open in list_coords:
                image_original.add_patch(Rectangle((x, y),w,h,
                                              edgecolor=(1, 0, 0) if closed else (0, 1, 0),
                                              facecolor='none',
                                              lw=3))
            for x, y, w, h, c, closed, open in list_coords:

                color = [1, 0, 0] if closed == 1 else [0, 1, 0]
                image_original.add_patch(Rectangle((x+w-57, y-32), 60, 32,
                                              fc=color + [0.3], ec=color + [1],
                                              linestyle='-',

                                              lw=1))

                image_original.text(x+w-45 if len(str(round(c, 2))) == 3 else x+w-54, y-8, round(c, 2), fontsize = 10)


            for (cx, cy, w, h, c, closed, open) in detected_bboxes_image.tolist():

                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)

                tasknet_prediction_unfiltered.add_patch(Rectangle((max(2,x), max(42, y)),min(x2-x, w_image-2),min(237, y2- y),
                                                   edgecolor=(1, 0, 0) if closed else (0, 1, 0),
                                                   facecolor='none',
                                                   lw=3))

            for (cx, cy, w, h, c, _, _), (background, closed, open) in zip(detected_bboxes_image.tolist(), labels_encoded_image.tolist()):

                if background == 1:
                    continue
                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)

                correct_predicitions_unfiltered.add_patch(Rectangle((max(2,x), max(42, y)),min(x2-x, w_image-2),min(237, y2- y),
                                                                  edgecolor=(1, 0, 0) if closed == 1 else (0, 1, 0),
                                                                  facecolor='none',
                                                                  lw=3))

            for (cx, cy, w, h, c, _, _), (background, closed, open) in zip(detected_bboxes_image.tolist(), labels_encoded_image.tolist()):

                if background != 1:
                    continue
                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)

                wrong_predicitions_unfiltered.add_patch(Rectangle((max(2,x), max(42, y)),min(x2-x, w_image-2),min(237, y2- y),
                                                                    edgecolor=(0, 0, 1),
                                                                    facecolor='none',
                                                                    lw=3))

            for (cx, cy, w, h, c, _, _), (background, closed, open) in zip(detected_bboxes_image.tolist(), labels_encoded_image.tolist()):


                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)

                if closed == 1:
                    color = (1, 0, 0)
                elif open == 1:
                    color = (0, 1, 0)
                else:
                    color = (0, 0, 1)
                all_predicitions_unfiltered.add_patch(Rectangle((max(2,x), max(42, y)),min(x2-x, w_image-2),min(237, y2- y),
                                                                  edgecolor=color,
                                                                  facecolor='none',
                                                                  lw=3))

            for (cx, cy, w, h, c, closed, open) in detected_bboxes_predicted_image.tolist():
                labels = [round(closed, 2), round(open, 2)]
                label = labels.index(max(labels))

                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)

                if label == 0:
                    color = (1, 0, 0)
                else:
                    color = (0, 1, 0)

                predicted_filternet.add_patch(Rectangle((max(2,x), max(42, y)),min(x2-x, w_image-2),min(237, y2- y),
                                                                edgecolor=color,
                                                                facecolor='none',
                                                                lw=3))
            fig.tight_layout()
            plt.savefig(save_path+ f'/{i}.png')
            i+=1
            #plt.show()
            plt.close()

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


