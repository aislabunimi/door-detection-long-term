from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.plot import plot_losses


model = FasterRCNN(model_name=FASTER_RCNN, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1_40_EPOCHS)
checkpoint = model.load_checkpoint()
print(checkpoint['logs'])
plot_losses(checkpoint['logs'], save_to_file=False)