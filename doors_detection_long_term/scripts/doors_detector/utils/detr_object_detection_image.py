import torch
import datasets as detr_dataset
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detection_long_term.doors_detector.models.detr import Detr, PostProcess
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def args():
    pass


args.coco_path = '/home/michele/myfiles/coco'
args.masks = False


if __name__ == '__main__':
    model = Detr(model_name=DETR_RESNET50, pretrained=True)
    model.eval()
    dataset = detr_dataset.coco.build('train', args)

    # Get first image from coco dataset
    img, target = dataset[0]
    img = img.unsqueeze(0)
    outputs = model(img)

    post_processor = PostProcess()
    img_size = list(img.size()[2:])
    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([img_size]))

    for image_data in processed_data:
        # keep only predictions with 0.7+ confidence

        keep = image_data['scores'] > 0.7

        # Show image with bboxes

        # Denormalize image tensor and convert to PIL image
        pil_image = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        plt.figure(figsize=(16, 10))

        plt.imshow(T.ToPILImage(mode='RGB')(pil_image[0]).convert("RGB"))
        ax = plt.gca()

        colors = COLORS * 100
        for label, score, (xmin, ymin, xmax, ymax), c in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep], colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            text = f'{CLASSES[label]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))

        plt.axis('off')
        plt.show()
