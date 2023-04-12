import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.utilities.utils import collate_fn, collate_fn_yolov5, \
    collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import torchvision.transforms as T


#train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.25, use_negatives=False)

train, test, labels, _ = get_final_doors_dataset_real_data('floor4', 0.25)
#print(f'Train set size: {len(train)}', f'Validation set size: {len(validation)}', f'Test set size: {len(test)}')
data_loader_train = DataLoader(train, batch_size=6, collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)
#data_loader_validation = DataLoader(validation, batch_size=6, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=6, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

COLORS = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

for d, data in enumerate(data_loader_train):
    images, targets, new_targets = data

    for count, (image, target) in enumerate(zip(images, new_targets)):
        plt.close()
        pil_image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.figure(figsize=(16, 10))
        t = T.ToPILImage()(pil_image)
        plt.imshow(t)

        ax = plt.gca()
        for box, label in zip(target['boxes'], target['labels']):
            x_min, y_min, x_max, y_max = box
            print(x_min)
            img_width, img_height = image.size()[1], image.size()[2]
            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color=COLORS[int(label)], linewidth=3))

        plt.axis('off')
        plt.show()