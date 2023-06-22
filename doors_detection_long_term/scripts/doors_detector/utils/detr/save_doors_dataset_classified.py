import random

import cv2
import numpy as np
import torch
import os
from matplotlib import pyplot as plt
import torchvision.transforms as T
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat, CoordinatesType
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything, collate_fn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import torch.nn.functional as F

params = {
    'seed': 0
}

path = '/media/antonazzi/hdd/classifications/trained_on_gd_day_tested_on_floor_1_night'

if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    # Create_folder
    if not os.path.exists(path):
        os.mkdir(path)

    #train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
    #train, test, labels, COLORS = get_final_doors_dataset(2, 'house1', train_size=0.25, use_negatives=False)
    #train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.75, use_negatives=True)
    train, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_25_EPOCHS_40)
    model.eval()

    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)


    total = 0
    for data in data_loader_test:
        images, targets = data

        outputs = model(images)

        """
        # Print real boxes
        outputs['pred_logits'] = torch.tensor([[[0, 1.0] for b in target['boxes']]], dtype=torch.float32)
        outputs['pred_boxes'] = target['boxes'].unsqueeze(0)
        """

        for image, target, pred_logits, pred_boxes in zip(images, targets, outputs['pred_logits'], outputs['pred_boxes']):
            # keep only predictions with 0.7+ confidence
            print(pred_logits)
            prob = F.softmax(pred_logits, -1)
            print(prob)
            print(prob[..., :-1])
            scores_image, labels_images = prob[..., :-1].max(-1)
            print(scores_image, labels_images)
            scores_image = scores_image.tolist()
            labels_images = labels_images.tolist()
            pred_boxes = pred_boxes.tolist()
            print(scores_image)
            #keep = scores_image > 0.75

            print(target)

            target_image = image.to('cpu')
            target_image = target_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            target_image = target_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            target_image = cv2.cvtColor(np.transpose(np.array(target_image), (1, 2, 0)), cv2.COLOR_RGB2BGR)

            detected_image = target_image.copy()
            image_size = image.size()[1:][::-1]

            for gt_box, label in zip(target['boxes'], target['labels']):
                x1, y1, w, h = gt_box.tolist()

                box = BoundingBox(
                    image_name=str(1),
                    class_id=str(label),
                    coordinates=(x1, y1, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                    confidence=1,
                    type_coordinates=CoordinatesType.RELATIVE,
                    img_size=image_size
                )
                x1, y1, w, h = box.get_absolute_bounding_box(BBFormat.XYWH)
                target_image = cv2.rectangle(target_image, (int((x1)), int((y1))),
                                               (int((x1 + w)), int((y1 + h))), COLORS[label], 2)

            #scores_image, pred_logits, pred_boxes
            for box_coords, score, label in zip(pred_boxes, scores_image, labels_images):
                x1, y1, w, h = box_coords
                box = BoundingBox(
                    image_name=str(1),
                    class_id=str(label),
                    coordinates=(x1, y1, w, h),
                    bb_type=BBType.DETECTED,
                    format=BBFormat.XYWH,
                    confidence=score,
                    type_coordinates=CoordinatesType.RELATIVE,
                    img_size=image_size
                )
                x1, y1, w, h = box.get_absolute_bounding_box(BBFormat.XYWH)
                print(x1, y1, w, h, box_coords, image_size)
                if box.get_confidence() >= 0.8:
                    total += 1


                    detected_image = cv2.rectangle(detected_image, (int((x1)), int((y1))),
                                                 (int((x1 + w)), int((y1 + h))), COLORS[int(label)], 2)

            show_image = cv2.hconcat([target_image, detected_image])

            cv2.imshow('show', show_image)
            cv2.waitKey()
    print(total)
            # Show image with bboxes
