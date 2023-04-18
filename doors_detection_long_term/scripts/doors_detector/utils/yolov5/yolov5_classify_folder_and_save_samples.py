import cv2
import torch
from PIL import Image
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import os
import torchvision.transforms as T


model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75)

model.to('cpu')
model.eval()
model.model.eval()

load_path = '/home/antonazzi/Downloads/images_floor4'
save_path = '/home/antonazzi/Downloads/floor_4_fine_tune_75'
if not os.path.exists(save_path):
    os.mkdir(save_path)

images_names = os.listdir(load_path)
images_names.sort()
images = [cv2.imread(os.path.join(load_path, file_name)) for file_name in images_names]
model.to('cuda')

transform = T.Compose([
    #T.RandomResize([std_size], max_size=max_size),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.Pad([0, 40]) # Check according image size
])


with torch.no_grad():
    for i, image in tqdm(enumerate(images), total=len(images_names)):

        new_img = transform(Image.fromarray(image[..., [2, 1, 0]])).unsqueeze(0).to('cuda')

        preds, train_out = model.model(new_img)
        #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
        preds = non_max_suppression(preds,
                                    0.25,
                                    0.45,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=300)

        img_size = list(new_img.size()[2:])

        for i, (image, boxes) in enumerate(zip(images, preds)):
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
