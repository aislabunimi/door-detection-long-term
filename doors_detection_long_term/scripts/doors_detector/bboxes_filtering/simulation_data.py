import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetsCreatorBBoxes, \
    Type
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 15

dataset_creator_bboxes = DatasetsCreatorBBoxes(num_bboxes=num_bboxes)

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']

train_student, validation_student, unlabelled_bbox_filter, test, labels, _ = get_final_doors_dataset_bbox_filter(folder_name='house1', train_size_student=.15)

data_loader_unlabelled = DataLoader(unlabelled_bbox_filter, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

model = YOLOv5Model(model_name=YOLOv5, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1_60_EPOCHS)

model.to('cuda')
model.model.eval()

with torch.no_grad():
    for images, targets, converted_boxes in tqdm(data_loader_unlabelled, total=len(data_loader_unlabelled)):
        images = images.to('cuda')
        preds, train_out = model.model(images)
        preds = non_max_suppression(preds,
                                    0.01,
                                    0.90,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=num_bboxes)


        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, Type.TRAINING)

    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test)):
        images = images.to('cuda')
        preds, train_out = model.model(images)
        preds = non_max_suppression(preds,
                                    0.01,
                                    0.90,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=num_bboxes)


        dataset_creator_bboxes.add_yolo_bboxes(images, targets, preds, Type.TEST)


dataset_creator_bboxes.filter_bboxes(iou_threshold=0.75, filter_multiple_detection=False, consider_label=False)
#dataset_creator_bboxes.visualize_bboxes(show_filtered=True)

train_bboxes, test_bboxes = dataset_creator_bboxes.create_datasets()

train_dataset_bboxes = DataLoader(train_bboxes, batch_size=4, collate_fn=collate_fn_bboxes, num_workers=4)

for data in train_dataset_bboxes:
    images, targets, converted_boxes, filtered = data
    images_opencv = []
    w_image, h_image = images.size()[2:][::-1]
    for image, target, bboxes, filter in zip(images, targets, converted_boxes, filtered):
        image = image.to('cpu')
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image_converted_bboxes = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
        target_image = image_converted_bboxes.copy()
        for (cx, cy, w, h, confidence, closed, open), f in zip(bboxes.tolist(), filter.tolist()):
            print(f)
            x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
            x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
            print(x, y, x2, y2)
            if f == 1:
                label = 0 if closed == 1 else 1
                image_converted_bboxes = cv2.rectangle(image_converted_bboxes, (x, y),
                                             (x2, y2), colors[label], 2)
        for box in target['gt_bboxes']:
            x, y, w, h = box.get_absolute_bounding_box()
            print(x, y, w, h)
            label = int(box.get_class_id())
            target_image = cv2.rectangle(target_image, (int(x), int(y)),
                                                   (int(x + w), int(y + h)), colors[label], 2)

        images_opencv.append(cv2.hconcat([target_image, image_converted_bboxes]))
    new_image = cv2.vconcat(images_opencv)
    cv2.imshow('show', new_image)
    cv2.waitKey()


