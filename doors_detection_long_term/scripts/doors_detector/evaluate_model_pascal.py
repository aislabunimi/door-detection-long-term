from src.evaluators.pascal_voc_evaluator import plot_precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.pascal_evaluator import PascalEvaluator
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.evaluators.coco_evaluator import CocoEvaluator
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything, collate_fn
from dataset_configurator import *
from doors_detection_long_term.doors_detector.models.detr_door_detector import *


device = 'cuda'
batch_size = 1

if __name__ == '__main__':
    seed_everything(0)

    train, test, labels, _ = get_final_doors_dataset(experiment=1, folder_name='house1', train_size=0.2, use_negatives=False)

    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1)

    model.eval()
    model.to(device)

    data_loader_test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, drop_last=False, num_workers=4)
    evaluator = PascalEvaluator()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
        images = images.to(device)
        outputs = model(images)
        evaluator.add_predictions(targets=targets, predictions=outputs)

    metrics = evaluator.get_metrics()
    plot_precision_recall_curve(metrics['per_class'])
    print(f'mAP = {metrics["mAP"]}')
    print(metrics['per_class']["0"].keys())
    for label, values in metrics['per_class'].items():
        print(f'Label {label} -> AP = {values["AP"]}, Total positives = {values["total positives"]}, TP = {values["total TP"]}, FP = {values["total FP"]}')

    #print(metrics['per_class']["0"]['precision'])