import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetsCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.bbox_filter_network import BboxFilterNetwork, TEST
from doors_detection_long_term.doors_detector.models.model_names import BBOX_FILTER_NETWORK, YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5 import YOLOv5Model, EXP_1_HOUSE_1_60_EPOCHS
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_bbox_filter

num_bboxes = 15
bbox_model = BboxFilterNetwork(num_bboxes=num_bboxes, model_name=BBOX_FILTER_NETWORK,
                               pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST)

bbox_model.to('cuda')

train_student, validation_student, unlabelled_bbox_filter, test, labels, _ = get_final_doors_dataset_bbox_filter(folder_name='house1', train_size_student=.15)

data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)
model = YOLOv5Model(model_name=YOLOv5, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1_60_EPOCHS)
model.to('cuda')
model.eval()
dataset_creator_bboxes = DatasetsCreatorBBoxes(num_bboxes=num_bboxes)

with torch.no_grad():
    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test)):
        images = images.to('cuda')
        preds, train_out = model.model(images)
        preds = non_max_suppression(preds,
                                    0.01,
                                    0.90,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=num_bboxes)


        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, ExampleType.TEST)

dataset_creator_bboxes.filter_bboxes(iou_threshold=0.75, filter_multiple_detection=False, consider_label=False)
#dataset_creator_bboxes.visualize_bboxes(show_filtered=True)

train_bboxes, test_bboxes = dataset_creator_bboxes.create_datasets()

#train_dataset_bboxes = DataLoader(train_bboxes, batch_size=4, collate_fn=collate_fn_bboxes, num_workers=4)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes, num_workers=4)
evaluator = MyEvaluator()
evaluator_complete_metric = MyEvaluatorCompleteMetric()
with torch.no_grad():
    bbox_model.eval()
    for data in tqdm(test_dataset_bboxes, total=len(test_dataset_bboxes), desc=f'TEST'):
        images, targets, converted_boxes, filtered = data
        #print(converted_boxes, filtered)
        images = images.to('cuda')
        converted_boxes = converted_boxes.to('cuda')
        filtered = filtered.to('cuda')

        preds = bbox_model(images, converted_boxes)
        print(preds)

