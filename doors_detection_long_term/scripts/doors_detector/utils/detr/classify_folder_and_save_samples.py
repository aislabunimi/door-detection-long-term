import random

import cv2
import numpy as np
import torch
from PIL import Image
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.dataset_manager import DatasetManager
from matplotlib import pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.door_sample_real_data import DoorSample
from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import os


model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR4_GIBSON_60_FINE_TUNE_75_EPOCHS_40)

transform = T.Compose([
    #T.RandomResize([std_size], max_size=max_size),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
load_path = '/home/antonazzi/Downloads/images_floor4'
save_path = '/home/antonazzi/Downloads/floor_4_fine_tune_25'

images_names = os.listdir(load_path)
images_names.sort()
#print(images_names)
for i, file_name in tqdm(enumerate(images_names[2000:]), total=len(images_names)):
    image = cv2.imread(os.path.join(load_path, file_name))

    new_img = transform(Image.fromarray(image[..., [2, 1, 0]])).unsqueeze(0)

    outputs = model(new_img)
    post_processor = PostProcess()
    img_size = list(new_img.size()[2:])

    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([img_size]))

    for image_data in processed_data:
        # keep only predictions with 0.7+ confidence

        keep = image_data['scores'] > 0.7

        save_image =image.copy()
        for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
            label = label.item()
            colors = {0: (0, 0, 255), 1: (0, 255, 0)}

            save_image = cv2.rectangle(save_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[label])
            #ax.text(xmin, ymin, text, fontsize=15,
            #bbox=dict(facecolor='yellow', alpha=0.5))
        cv2.imwrite(os.path.join(save_path, f'image_{i}.jpg'), save_image)
