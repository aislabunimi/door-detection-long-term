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
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import seed_everything, collate_fn

from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import torch.nn.functional as F

params = {
    'seed': 0
}

path = '/mnt/54685f13-5e79-4e84-b2a6-bf3a0eba4d7f/classifications/dataset_gd'

if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    # Create_folder
    if not os.path.exists(path):
        os.mkdir(path)

    #train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
    #train, test, labels, COLORS = get_final_doors_dataset(2, 'house1', train_size=0.25, use_negatives=False)
    #train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.75, use_negatives=True)
    train, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name='floor4', train_size=0.75)

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    #model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_25_EPOCHS_40)
    #model.eval()

    data_loader_train = DataLoader(train, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)


    total = 0
    for c, data in enumerate(train + test):
        _, _, door_example = data
        image = door_example.get_bgr_image()
        bboxes = door_example.get_bounding_boxes()

        for label, x1, y1, w, h in bboxes:

            x2, y2 = x1+w, y1+h
            if x1 == 0:
                x1+=1
            if y1 == 0:
                y1+=1
            if x2 == image.shape[1]:
                x2 -= 2
            if y2 == image.shape[0]:
                y2-=2

            image = cv2.rectangle(image, (x1, y1), (x2 , y2), (0,255,0) if label == 1 else (0, 0, 255), 2)

        cv2.imwrite(path + f"/{c}.png", image)
