import numpy as np

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.background_grid_network import IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL
from doors_detection_long_term.doors_detector.models.bbox_filter_network_geometric import *
from doors_detection_long_term.doors_detector.models.model_names import BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND

grid_dim = [(2**i, 2**i) for i in range(3, 6)][::-1]

model = BboxFilterNetworkGeometricBackground(initial_channels=7, image_grid_dimensions=grid_dim, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC_BACKGROUND, pretrained=True, grid_network_pretrained=True, dataset_name=FINAL_DOORS_DATASET,
                                                  description=IMAGE_NETWORK_GEOMETRIC_BACKGROUND, description_background=IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL)
model.set_description(globals()[f'IMAGE_NETWORK_GEOMETRIC_BACKGROUND_GIBSON_DD2_FINE_TUNE_FLOOR1_75_BBOX_100'.upper()])
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