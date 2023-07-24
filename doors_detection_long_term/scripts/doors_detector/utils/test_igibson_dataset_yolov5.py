import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from torch.utils.data import DataLoader
import os

from doors_detection_long_term.doors_detector.utilities.utils import collate_fn, collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import torchvision.transforms as T

train, validation, labels, _ = get_igibson_dataset_scene("Rs_int")
print(f"Train dataset size: {len(train)} | Validation dataset size: {len(validation)}")
data_loader_train = DataLoader(train, batch_size=16, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)
data_loader_validation = DataLoader(validation, batch_size=8, collate_fn=collate_fn_yolov5, shuffle=False, num_workers=4)

COLORS = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
save_path = "/media/michele/Elements/dataloader_test"

fig = plt.figure(figsize=(16, 16))
grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)

train_save_path = os.path.join(save_path, "train")
if not os.path.exists(train_save_path):
    os.makedirs(train_save_path)

for data_index, data in enumerate(data_loader_train):
    images, targets, boxes = data
    #print(data_index, len(images), len(targets), len(boxes))

    for image_index, image in enumerate(images):
        ax = grid[image_index]
        ax.axis("off")

        #plt.close()
        pil_image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        #plt.figure(figsize=(16, 10))
        t = T.ToPILImage()(pil_image)
        #plt.imshow(t)
        ax.imshow(t)

        #ax_plot = plt.gca()

        image_boxes = list(filter(lambda b: b[0].item() == image_index, boxes)) # currente image boxes

        for img, label, x, y, width, height in image_boxes:
            img, label, x, y, width, height = img.item(), label.item(), x.item(), y.item(), width.item(), height.item()
            img_width, img_height = image.size()[1], image.size()[2]

            ax.add_patch(plt.Rectangle(
                ((x - width/2) * img_width, (y - height/2)*img_height), # box top-left corner
                width * img_width, height * img_height,                 # box width and height
                fill=False, color=COLORS[int(label)], linewidth=3       # box style
            ))

    #plt.show()
    batch_save_path = os.path.join(train_save_path, "batch_"+str(data_index)+".png")
    print("saving", batch_save_path)
    fig.savefig(batch_save_path, dpi=fig.dpi)
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
    #break


test_save_path = os.path.join(save_path, "test")
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

for data_index, data in enumerate(data_loader_validation):
    images, targets, boxes = data

    for image_index, image in enumerate(images):
        ax = grid[image_index]
        ax.axis("off")

        pil_image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        t = T.ToPILImage()(pil_image)
        ax.imshow(t)

        image_boxes = list(filter(lambda b: b[0].item() == image_index, boxes)) # currente image boxes

        for img, label, x, y, width, height in image_boxes:
            img, label, x, y, width, height = img.item(), label.item(), x.item(), y.item(), width.item(), height.item()
            img_width, img_height = image.size()[1], image.size()[2]

            ax.add_patch(plt.Rectangle(
                ((x - width/2) * img_width, (y - height/2)*img_height), # box top-left corner
                width * img_width, height * img_height,                 # box width and height
                fill=False, color=COLORS[int(label)], linewidth=3       # box style
            ))

    #plt.show()
    batch_save_path = os.path.join(test_save_path, "batch_"+str(data_index)+".png")
    print("saving", batch_save_path)
    fig.savefig(batch_save_path, dpi=fig.dpi)
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)
    #break