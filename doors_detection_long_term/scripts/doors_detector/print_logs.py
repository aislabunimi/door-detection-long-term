from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.plot import plot_losses


model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_HOUSE_1_25)
checkpoint = model.load_checkpoint()

plot_losses(checkpoint['logs'])