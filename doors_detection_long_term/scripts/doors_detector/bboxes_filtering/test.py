import torch
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.BboxFilterNetwork import BboxFilterNetwork, TEST
from doors_detection_long_term.doors_detector.models.model_names import BBOX_FILTER_NETWORK
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import \
    get_final_doors_dataset_real_data_bbox_filter


params = {
    'batch_size': 6
}
num_boxes = 15
model = BboxFilterNetwork(BBOX_FILTER_NETWORK, pretrained=True, num_bboxes=num_boxes, dataset_name=FINAL_DOORS_DATASET, description=TEST)
print(model)
train, control, fine_tune, test, labels, COLORS = get_final_doors_dataset_real_data_bbox_filter(folder_name='floor1', train_size=0.15, control_size=0.15, fine_tune_size=0.45, test_size=0.25)

data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

for images, targets, converted_boxes in data_loader_train:
    print(images.size())
    o = model(images, torch.ones((params['batch_size'], num_boxes, 6)))
    print(o.size())
