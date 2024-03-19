import os

import cv2
import torch.optim
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
from PIL import Image
import torchvision.transforms as T

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 20

house = 'house_matteo'
folder = f"/home/antonazzi/myfiles/{house}"

images = sorted(os.listdir(folder))
print(images)
print(images)


dataset_creator_bboxes = DatasetCreatorBBoxes()
dataset_creator_bboxes.set_folder_name(f'faster_rcnn_general_detector_gibson_dd2_{house}_bag')

model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_GENERAL_DETECTOR_GIBSON_DEEP_DOORS_2_60_EPOCHS,
                   box_score_thresh=0.0, box_nms_thresh=1.0, box_detections_per_img=300)
model.to('cuda')
model.model.eval()
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.Pad([0, 40]) # Check according image size
])
with torch.no_grad():
    for file in tqdm(images):

        image = cv2.imread(os.path.join(folder, file))
        img = transform(Image.fromarray(image[..., [2, 1, 0]])).unsqueeze(0).to('cuda')
        images = img.to('cuda')
        preds = model.model(images)
        for pred in preds:
            pred['labels'] = pred['labels']-1

        dataset_creator_bboxes.add_faster_rcnn_bboxes(images, ({'labels':torch.tensor([]), 'boxes' :torch.tensor([])},), preds, ExampleType.TRAINING)

    dataset_creator_bboxes.export_dataset()