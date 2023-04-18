from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

train, test, _, _ = get_final_doors_dataset_all_envs()
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

model = FasterRCNN(model_name=FASTER_RCNN, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_GENERAL_DETECTOR_GIBSON_60_EPOCHS)

model.eval()
model.to('cuda')

with torch.no_grad():
    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test), desc='classifying'):
        images = images.to('cuda')
        output = model.model(images)
        #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
        print(output)