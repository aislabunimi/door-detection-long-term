import cv2
from matplotlib import pyplot as plt
import torchvision.transforms as T
from models.detr import SetCriterion
from models.matcher import HungarianMatcher
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import seed_everything, collate_fn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

params = {
    'epochs': 40,
    'batch_size': 1,
    'seed': 0,
    'lr': 1e-5,
    'weight_decay': 1e-4,
    'lr_drop': 20,
    'lr_backbone': 1e-6,
    # Criterion
    'bbox_loss_coef': 5,
    'giou_loss_coef': 2,
    'eos_coef': 0.1,
    # Matcher
    'set_cost_class': 1,
    'set_cost_bbox': 5,
    'set_cost_giou': 2,
}

path = '/home/antonazzi/Downloads/detr_gd'

if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    # Create_folder
    if not os.path.exists(path):
        os.mkdir(path)

    #train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
    #train, test, labels, COLORS = get_final_doors_dataset(2, 'house1', train_size=0.25, use_negatives=False)
    train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=2, folder_name='house21', train_size=0.75, use_negatives=False)
    #train, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
    data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)
    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_21_2_LAYERS_BACKBONE_60_EPOCHS)
    model.eval()
    losses = ['labels', 'boxes', 'cardinality']
    weight_dict = {'loss_ce': 1, 'loss_bbox': params['bbox_loss_coef'], 'loss_giou': params['giou_loss_coef']}
    matcher = HungarianMatcher(cost_class=params['set_cost_class'], cost_bbox=params['set_cost_bbox'], cost_giou=params['set_cost_giou'])
    criterion = SetCriterion(len(labels.keys()), matcher=matcher, weight_dict=weight_dict,
                             eos_coef=params['eos_coef'], losses=losses)

    for i, training_data in enumerate(data_loader_train):
        images, targets = training_data
        images = images.to('cuda')

        # Move targets to device
        targets = [{k: v.to('cuda') for k, v in target.items() if k != 'folder_name' and k != 'absolute_count'} for target in targets]

        outputs = model(images)
        # Compute losses
        losses_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # Losses are weighted using parameters contained in a dictionary
        losses = sum(losses_dict[k] * weight_dict[k] for k in losses_dict.keys() if k in weight_dict)
        """
        # Print real boxes
        outputs['pred_logits'] = torch.tensor([[[0, 1.0] for b in target['boxes']]], dtype=torch.float32)
        outputs['pred_boxes'] = target['boxes'].unsqueeze(0)
        """

        post_processor = PostProcess()
        img_size = list(images.size()[2:])
        processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([img_size]))

        for image_data in processed_data:
            # keep only predictions with 0.7+ confidence

            keep = image_data['scores'] > 0.75
            pil_image = images[0] * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            pil_image = pil_image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            open_cv_image = np.array(pil_image)
            # Convert RGB to BGR
            save_image = open_cv_image[:, :, ::-1].copy()
            # Show image with bboxes

            for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
                label = label.item()
                colors = {0: (0, 0, 255), 1: (0, 255, 0)}
                save_image = cv2.rectangle(save_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[label], 2)

            cv2.imwrite(os.path.join(path, 'image_{0:05d}.png'.format(i)), save_image)