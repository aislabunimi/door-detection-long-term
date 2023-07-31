import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import seed_everything, collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import os
import torchvision.transforms as T

#train, test, labels, _ = get_final_doors_dataset_real_data(folder_name='chemistry_floor0', train_size=0.25)
train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, train_size=0.25, folder_name='house1')
print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')
#data_loader_validation = DataLoader(validation, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1_60_EPOCHS)

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
    for i in range(len(test)):
        img, target, door_sample = test[i]
        img = transform(img).unsqueeze(0).to('cuda')

        preds, train_out = model.model(img)
        #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
        preds = non_max_suppression(preds,
                                    0.01,
                                    0.90,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=300)
        img_size = list(img.size()[2:])
        print(preds)
        for count, (image, boxes) in enumerate(zip(img, preds)):
            # keep only predictions with 0.7+ confidence
            save_image =door_sample.get_bgr_image().copy()

            img_size = save_image.shape
            #save_image =  cv2.copyMakeBorder(save_image, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            for x1, y1, x2, y2, conf, label in boxes:
                label, x1, y1, x2, y2 = label.item(), x1.item(), y1.item(), x2.item(), y2.item()
                x1 = int(min(img_size[1], max(.0, x1)))
                y1 = int(min(img_size[0], max(.0, y1 - padding_height)))
                x2 = int(min(img_size[1], max(.0, x2)))
                y2 = int(min(img_size[0], max(.0, y2 - padding_height)))
                colors = {0: (0, 0, 255), 1: (0, 255, 0)}
                save_image = cv2.rectangle(save_image, (x1, y1), (x2, y2), colors[label], 2)
                #ax.text(xmin, ymin, text, fontsize=15,
                #bbox=dict(facecolor='yellow', alpha=0.5))
            cv2.imwrite(os.path.join(save_path, 'image_{0:05d}.png'.format(i)), save_image)
