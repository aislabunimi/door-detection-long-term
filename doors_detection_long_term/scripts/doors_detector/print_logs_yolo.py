from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50, YOLOv5
from doors_detection_long_term.doors_detector.utilities.plot import plot_losses


model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75)
checkpoint = model.load_checkpoint()
print(checkpoint['logs'])
plot_losses(checkpoint['logs'], save_to_file=True)