import random

import cv2
import numpy as np
import torch
from PIL import Image
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.dataset_manager import DatasetManager
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample_real_data import DoorSample
from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from dataset_configurator import *

folder_manager = DatasetFolderManager(dataset_path=real_final_doors_dataset_path, folder_name='chemistry_floor0', sample_class=DoorSample)

model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_CHEMISTRY_FLOOR0_GIBSON_60_FINE_TUNE_75_EPOCHS_40)

transform = T.Compose([
    #T.RandomResize([std_size], max_size=max_size),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
path = '/home/antonazzi/myfiles/classified2'
for i in folder_manager.get_samples_absolute_counts(label=1):
    door_sample = folder_manager.load_sample_using_absolute_count(i, use_thread=False)

    new_img = transform(Image.fromarray(door_sample.get_bgr_image()[..., [2, 1, 0]])).unsqueeze(0)

    outputs = model(new_img)
    post_processor = PostProcess()
    img_size = list(new_img.size()[2:])

    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([img_size]))

    for image_data in processed_data:
        # keep only predictions with 0.7+ confidence

        keep = image_data['scores'] > 0.75

        save_image = door_sample.get_bgr_image().copy()
        for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
            print((xmin, ymin, xmax, ymax))
            label = label.item()
            colors = {0: (0, 0, 255), 1: (0, 255, 0)}

            save_image = cv2.rectangle(save_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[label])
            #ax.text(xmin, ymin, text, fontsize=15,
            #bbox=dict(facecolor='yellow', alpha=0.5))

        cv2.imwrite(path + f'/zclass_{i}.png', save_image)
