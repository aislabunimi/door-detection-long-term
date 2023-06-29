import cv2
import torch
from PIL import Image
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat, CoordinatesType
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything, collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import os
import torchvision.transforms as T

train, test, labels, _ = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
#train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, train_size=0.25, folder_name='house1')
#print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')
#data_loader_validation = DataLoader(validation, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15)

model.to('cpu')
model.eval()

model.model.eval()

save_path = '/home/antonazzi/Downloads/test_yolo_gd'

if not os.path.exists(save_path):
    os.mkdir(save_path)

model.to('cuda')
padding_height = 0
padding_width = 0
transform = T.Compose([
    T.Pad([padding_width, padding_height]) # Check according image size
])

with torch.no_grad():
    for images, targets, converted_bboxes in data_loader_test:
        images = images.to('cuda')

        preds, train_out = model.model(images)
        #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
        preds = non_max_suppression(preds,
                                    0.75,
                                    0.5,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=300)
        for count, (image, target, boxes) in enumerate(zip(images, targets, preds)):
            # keep only predictions with 0.7+ confidence
            target_image = image.to('cpu')
            target_image = target_image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            target_image = target_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            target_image = cv2.cvtColor(np.transpose(np.array(target_image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            detected_image = target_image.copy()
            #save_image =  cv2.copyMakeBorder(save_image, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            for x1, y1, x2, y2, conf, label in boxes.tolist():
                x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                colors = {0: (0, 0, 255), 1: (0, 255, 0)}
                detected_image = cv2.rectangle(detected_image, (x1, y1), (x2, y2), colors[label], 2)
                #ax.text(xmin, ymin, text, fontsize=15,
                #bbox=dict(facecolor='yellow', alpha=0.5))
            #cv2.imwrite(os.path.join(save_path, 'image_{0:05d}.png'.format(i)), save_image)
            image_size = image.size()[1:][::-1]
            #for _, label, x1, y1, w, h in converted_bboxes.tolist():
            for gt_box, label in zip(target['boxes'], target['labels']):
                print(target['size'])
                x1, y1, w, h = gt_box.tolist()
                label = int(label.item())
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
                                             (int((x1 + w)), int((y1 + h))), colors[label], 2)
            image = cv2.hconcat([target_image, detected_image])
            cv2.imshow('show', image)
            cv2.waitKey()
