import cv2
from matplotlib import pyplot as plt
import torchvision.transforms as T
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

params = {
    'seed': 0
}

path = '/media/antonazzi/hdd/classifications/test_detr_opencv'

if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    # Create_folder
    if not os.path.exists(path):
        os.mkdir(path)

    #train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
    #train, test, labels, COLORS = get_final_doors_dataset(2, 'house1', train_size=0.25, use_negatives=False)
    #train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.75, use_negatives=True)
    train, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_25_EPOCHS_40)
    model.eval()

    for i in range(len(test)):
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

            keep = image_data['scores'] > 0.7

            # Show image with bboxes
            save_image =door_sample.get_bgr_image().copy()
            for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
                label = label.item()
                colors = {0: (0, 0, 255), 1: (0, 255, 0)}
                save_image = cv2.rectangle(save_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), colors[label])

            cv2.imwrite(os.path.join(path, 'image_{0:05d}.png'.format(i)), save_image)