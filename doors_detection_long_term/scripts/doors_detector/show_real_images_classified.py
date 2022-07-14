import os

import cv2
import numpy as np
import torchvision.transforms as T
from matplotlib import pyplot as plt

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
torch.set_grad_enabled(False)
path = '/home/michele/myfiles/real_images'

labels = {0: 'Closed door', 1: 'Open door'}
COLORS = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)

images = []
model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

for file in [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]:
    image = transform(cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)).unsqueeze(0)

    outputs = model(image)

    post_processor = PostProcess()
    img_size = list(image.size()[2:])
    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([img_size]))

    for image_data in processed_data:
        # keep only predictions with 0.7+ confidence

        keep = image_data['scores'] > 0.5

        # Show image with bboxes

        # Denormalize image tensor and convert to PIL image
        pil_image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.figure(figsize=(16, 10))

        plt.imshow(T.ToPILImage()(pil_image[0]))
        ax = plt.gca()

        for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
            label = label.item()
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=COLORS[label], linewidth=3))
            text = f'{labels[int(label)]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.show()





