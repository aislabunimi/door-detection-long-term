import cv2
import torch
from PIL import Image
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

train, test, labels, _ = get_final_doors_dataset_real_data(folder_name='floor4', train_size=0.25,)
print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')
data_loader_train = DataLoader(train, batch_size=3, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)
#data_loader_validation = DataLoader(validation, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)

model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75)

model.to('cpu')
model.eval()

model.model.eval()

save_path = '/home/antonazzi/Downloads/test_yolo'

model.to('cuda')

transform = T.Compose([
    T.Pad([0, 40]) # Check according image size
])

with torch.no_grad():
    for i in range(len(test)):
        img, target, door_sample = test[i]
        img = transform(img).unsqueeze(0)

        preds, train_out = model.model(img)
        #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
        preds = non_max_suppression(preds,
                                    0.25,
                                    0.45,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=300)
        img_size = list(img.size()[2:])

        for i, (image, boxes) in enumerate(zip(img, preds)):
            # keep only predictions with 0.7+ confidence
            save_image =image.copy()
            for x1, y1, x2, y2, conf, label in boxes:
                label, x1, y1, x2, y2 = label.item(), x1.item(), y1.item(), x2.item(), y2.item()
                x1 = int(min(img_size[0], max(.0, x1)))
                y1 = int(min(img_size[0], max(.0, y1)))
                x2 = int(min(img_size[0], max(.0, x2)))
                y2 = int(min(img_size[0], max(.0, y2)))
                colors = {0: (0, 0, 255), 1: (0, 255, 0)}
                save_image = cv2.rectangle(save_image, (x1, y1), (x2, y2), colors[label])
                #ax.text(xmin, ymin, text, fontsize=15,
                #bbox=dict(facecolor='yellow', alpha=0.5))
            cv2.imwrite(os.path.join(save_path, 'image_{0:05d}.png'.format(i)), save_image)
