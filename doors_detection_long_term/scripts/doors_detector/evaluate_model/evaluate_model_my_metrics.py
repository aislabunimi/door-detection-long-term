from src.evaluators.pascal_voc_evaluator import plot_precision_recall_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET, DEEP_DOORS_2_LABELLED
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
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

    #train, test, labels, COLORS = get_final_doors_dataset(experiment=2, folder_name='house1', train_size=0.25, use_negatives=True)
    #train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
    train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.25, use_negatives=True)


    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1_EPOCHS_ANALYSIS)

    model.eval()
    model.to(device)

    data_loader_test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, drop_last=False, num_workers=4)
    evaluator = MyEvaluator()

    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Evaluate model'):
        images = images.to(device)
        outputs = model(images)
        evaluator.add_predictions(targets=targets, predictions=outputs)

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.5, door_no_door_task=False, plot_curves=True, colors=COLORS)
    mAP = 0
    print('Results per bounding box:')
    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        mAP += values['AP']
        print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
        print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
    print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')

    mAP = 0
    print('Results per image')
    for label, values in sorted(metrics['per_image'].items(), key=lambda v: v[0]):
        print(f'\tLabel {label} -> Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}, FN = {values["FN"]}')
        print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')