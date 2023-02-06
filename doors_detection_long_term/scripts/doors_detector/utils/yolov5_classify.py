import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5 import YOLOv5Model, EXP_1_HOUSE_1
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn, collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis
import torchvision.transforms as T


train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.25, use_negatives=False)
print(f'Train set size: {len(train)}', f'Validation set size: {len(validation)}', f'Test set size: {len(test)}')
data_loader_train = DataLoader(train, batch_size=3, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)
data_loader_validation = DataLoader(validation, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1)

model.to('cpu')
model.eval()
model.model.eval()
COLORS = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

for d, data in enumerate(data_loader_test):
    images, targets, converted_boxes = data

    #output = model(images)
    preds, train_out = model.model(images)
    #print(preds.size(), train_out[0].size(), train_out[1].size(), train_out[2].size())
    preds = non_max_suppression(preds,
                                0.25,
                                0.45,

                                multi_label=True,
                                agnostic=True,
                                max_det=300)

    #print(preds)
    for i, (image, boxes) in enumerate(zip(images, preds)):

        plt.close()
        pil_image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.figure(figsize=(16, 10))
        t = T.ToPILImage()(pil_image)
        plt.imshow(t)

        ax = plt.gca()

        for x1, y1, x2, y2, conf, label in boxes:
            label, x1, y1, x2, y2 = label.item(), x1.item(), y1.item(), x2.item(), y2.item()
            x1 = min(256.0, max(.0, x1))
            y1 = min(256.0, max(.0, y1))
            x2 = min(256.0, max(.0, x2))
            y2 = min(256.0, max(.0, y2))
            print(x1, y1, x2, y2)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, color=COLORS[int(label)], linewidth=3))

        plt.axis('off')
        plt.savefig(f'ciao{d}.svg')
