import random

import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from dataset_configurator import *

params = {
    'seed': 0
}


if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    train, test, labels, COLORS = get_final_doors_dataset(experiment=1, folder_name='house1', train_size=0.8, use_negatives=True)

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1)
    model.eval()

    for i in range(10, 50):
        img, target, door_sample = test[i]
        img = img.unsqueeze(0)
        outputs = model(img)

        """
        # Print real boxes
        outputs['pred_logits'] = torch.tensor([[[0, 1.0] for b in target['boxes']]], dtype=torch.float32)
        outputs['pred_boxes'] = target['boxes'].unsqueeze(0)
        """

        post_processor = PostProcess()
        img_size = list(img.size()[2:])
        processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([img_size]))

        for image_data in processed_data:
            # keep only predictions with 0.7+ confidence

            keep = image_data['scores'] > 0.2

            # Show image with bboxes

            # Denormalize image tensor and convert to PIL image
            pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            plt.figure(figsize=(16, 10))

            plt.imshow(T.ToPILImage()(pil_image[0]))
            ax = plt.gca()

            for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                           fill=False, color=COLORS[label], linewidth=3))
                text = f'{labels[int(label)]}: {score:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15,
                        bbox=dict(facecolor='yellow', alpha=0.5))

            plt.axis('off')
            plt.show()