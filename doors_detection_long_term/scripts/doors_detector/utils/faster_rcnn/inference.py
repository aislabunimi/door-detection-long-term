import time

import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import torchvision.transforms as T

device = 'cuda'

#train, test, _, _ = get_final_doors_dataset_all_envs()
train, test, labels, _ = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
#train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, train_size=0.25, folder_name='house20')
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=EXP_GENERAL_DETECTOR_GIBSON_DEEP_DOORS_2_60_EPOCHS)
model.eval()
model.to(device)
padding_height = 40
padding_width = 0
transform = T.Compose([
    T.Pad([padding_width, padding_height]) # Check according image size
])

times = []
with torch.no_grad():
    for i in range(len(test)):
        img, target, door_sample = test[i]
        img = transform(img).unsqueeze(0).to(device)
        t = time.time()
        outputs = model.model(img)
        times.append(time.time() - t)

print( f'{1 / np.array(times[2:]).mean()} FPS in {device}')

