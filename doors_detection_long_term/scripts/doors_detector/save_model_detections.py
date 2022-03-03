import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as T
import os
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detector.utilities.utils import seed_everything
from dataset_configurator import get_final_doors_dataset

house = 'house1'



params = {
    'seed': 0
}

seed_everything(params['seed'])

houses = {
    'house1': [EXP_1_HOUSE_1, EXP_2_HOUSE_1_25, EXP_2_HOUSE_1_50, EXP_2_HOUSE_1_75],
    'house2': [EXP_1_HOUSE_2, EXP_2_HOUSE_2_25, EXP_2_HOUSE_2_50, EXP_2_HOUSE_2_75],
    'house7': [EXP_1_HOUSE_7, EXP_2_HOUSE_7_25, EXP_2_HOUSE_7_50, EXP_2_HOUSE_7_75],
    'house9': [EXP_1_HOUSE_9, EXP_2_HOUSE_9_25, EXP_2_HOUSE_9_50, EXP_2_HOUSE_9_75],
    'house10': [EXP_1_HOUSE_10, EXP_2_HOUSE_10_25, EXP_2_HOUSE_10_50, EXP_2_HOUSE_10_75],
    'house13': [EXP_1_HOUSE_13, EXP_2_HOUSE_13_25, EXP_2_HOUSE_13_50, EXP_2_HOUSE_13_75],
    'house15': [EXP_1_HOUSE_15, EXP_2_HOUSE_15_25, EXP_2_HOUSE_15_50, EXP_2_HOUSE_15_75],
    'house20': [EXP_1_HOUSE_20, EXP_2_HOUSE_20_25, EXP_2_HOUSE_20_50, EXP_2_HOUSE_20_75],
    'house21': [EXP_1_HOUSE_21, EXP_2_HOUSE_21_25, EXP_2_HOUSE_21_50, EXP_2_HOUSE_21_75],
    'house22': [EXP_1_HOUSE_22, EXP_2_HOUSE_22_25, EXP_2_HOUSE_22_50, EXP_2_HOUSE_22_75],
}

for house in houses.keys():
    save_path = '/media/antonazzi/hdd/model_detections/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, house)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train, test, labels, COLORS = get_final_doors_dataset(2, house, train_size=0.25, use_negatives=True)
    models = [
        (DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=model), title)
        for model, title in zip(houses[house],  ['GD', 'QD25', 'QD50', 'QD75'])
    ]

    for i, (img, target, door_sample) in enumerate(test):
        if len(target['boxes']) == 0:
            continue

        img = img.unsqueeze(0)
        outputs = [(model(img), title) for model, title in models]

        post_processor = PostProcess()
        img_size = list(img.size()[2:])
        processed_data_models = [(post_processor(outputs=output, target_sizes=torch.tensor([img_size])), title) for output, title in outputs]

        #pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        #pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        #fig, axes = plt.subplots(nrows=2, ncols=2)

        imgs = [door_sample.get_bgr_image().copy() for _ in range(4)]
        for img, post_processed_data in zip(imgs, processed_data_models):
            image_data, title = post_processed_data[0][0], post_processed_data[1]
            keep = image_data['scores'] > 0.5
            for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
                label = label.item()

                (xmin, ymin, xmax, ymax) = np.rint(np.array([xmin, ymin, xmax, ymax])).astype('int32')
                img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(np.array(COLORS[label])*255)[::-1].astype('int16').tolist(), thickness=2)
        r1 = np.concatenate((imgs[0], imgs[1]), axis=0)
        r2 = np.concatenate((imgs[2], imgs[3]), axis=0)
        final_image = np.concatenate((r1, r2), axis=1)
        #cv2.imwrite(save_path + f'/{i}.pdf', final_image)
        cv2.imwrite(save_path + f'/{i}.png', final_image)
        """for axis, post_processed_data in zip(axes.flatten(), processed_data_models):

            axis.imshow(T.ToPILImage()(pil_image[0]),cmap='',interpolation='')

            image_data, title = post_processed_data[0][0], post_processed_data[1]
            keep = image_data['scores'] > 0.5
            for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
                label = label.item()
                axis.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color=COLORS[label], linewidth=1))
                text = f'{labels[int(label)]}: {score:0.2f}'
                #axis.text(xmin, ymin - 2, text, fontsize=5)
                #axis.set_title(title)

            axis.axis('off')
        
        fig.savefig(save_path + f'/{i}.pdf')
        fig.savefig(save_path + f'/{i}.png')
        """



