import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetLoaderBBoxes import DatasetLoaderBBoxes
from doors_detection_long_term.doors_detector.models.background_grid_network import *
from doors_detection_long_term.doors_detector.models.bbox_filter_network_geometric import *
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_bboxes
colors = {0: (0, 0, 255), 1: (0, 255, 0)}


save_path = "/home/antonazzi/myfiles/save_bbox_filtering"

house = 'floor1'
quantity = 0.5
num_bboxes = 50
iou_threshold_matching = 0.5
confidence_threshold_original = 0.75
grid_dim = [(2**i, 2**i) for i in range(3, 6)][::-1]

dataset_loader_bboxes = DatasetLoaderBBoxes(folder_name=f'faster_rcnn_general_detector_gibson_dd2_{house}_{quantity}')
train_bboxes, test_bboxes = dataset_loader_bboxes.create_dataset(max_bboxes=num_bboxes, iou_threshold_matching=iou_threshold_matching, apply_transforms_to_train=True, shuffle_boxes=False)

test_dataset_bboxes = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes(use_confidence=True, image_grid_dimensions=grid_dim), num_workers=4)

bbox_model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=True, grid_network_pretrained=True, dataset_name=FINAL_DOORS_DATASET,
                                                  description=IMAGE_NETWORK_GEOMETRIC_BACKGROUND, description_background=IMAGE_NETWORK_GEOMETRIC_BACKGROUND_GIBSON_DD2_FINE_TUNE_FLOOR1_50_BBOX_30)


for data in test_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, target_boxes_grid, detected_boxes_grid = data
    detected_bboxes = detected_bboxes.transpose(1, 2)
    detected_bboxes = bbox_filtering_nms(detected_bboxes, confidence_threshold=confidence_threshold_original, iou_threshold=0.5, img_size=images.size()[::-1][:2])

    for image, detected_bboxes_image in zip(images, detected_bboxes):
        w_image, h_image = image.size()[1:][::-1]
        image = image.to('cpu')
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        image = np.transpose(np.array(image), (1, 2, 0))[:280,:]
        image[0:40, :] = [255, 255, 255]



        fig, (image_original, ax2) = plt.subplots(1, 2)
        image_original.imshow(image)
        image_original.axis('off')

        list_coords = []
        for (cx, cy, w, h, c, closed, open) in detected_bboxes_image.tolist():

            #print(cx, cy, w, h)
            x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
            x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
            list_coords.append([max(42, x), max(42, y),min(x2-x, w_image-2),min(237, y2- y), c, closed, open])
        for x, y, w, h, c, closed, open in list_coords:
            image_original.add_patch(Rectangle((x, y),w,h,
                                          edgecolor=(1, 0, 0) if closed else (0, 1, 0),
                                          facecolor='none',
                                          lw=4))
        for x, y, w, h, c, closed, open in list_coords:

            color = [1, 0, 0] if closed == 1 else [0, 1, 0]
            image_original.add_patch(Rectangle((x+w-57, y-32), 57, 32,
                                          fc=color + [0.3], ec=color + [1],
                                          linestyle='--',
                                          
                                          lw=2))

            image_original.text(x+w-46 if len(str(round(c, 2))) == 3 else x+w-53, y-8, round(c, 2), fontsize = 11)


        plt.show()
        plt.close()



