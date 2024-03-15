import numpy as np

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.plot import plot_losses


model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75)
checkpoint = model.load_checkpoint()
times = np.array(checkpoint['logs']['time'])
total = 0
sum = 0
print(times)
for t in times:
    total += t - sum
    sum = t
print(total, total / 60)
plot_losses(checkpoint['logs'], save_to_file=False)