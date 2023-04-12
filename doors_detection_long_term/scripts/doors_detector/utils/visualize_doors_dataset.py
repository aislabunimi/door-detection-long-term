import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

params = {
    'seed': 0
}


if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
    #train, test, labels, COLORS = get_final_doors_dataset(2, 'house1', train_size=0.25, use_negatives=False)
    #train, validation, labels, COLORS = get_final_doors_dataset_all_envs()

    for i in range(80, 140):
        img, target, door_sample = train[i]

        cv_image = door_sample.get_bgr_image()

        for [label, x, y, width, height] in door_sample.get_bounding_boxes():
            cv_image = cv2.rectangle(cv_image, (x, y), (x + width, y + height), (0, 255, 0) if label == 1 else (0, 0, 255), 2)

        cv2.imshow('image', cv_image)
        cv2.waitKey()


