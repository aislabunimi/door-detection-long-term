from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

house = 'house1'
train, validation, labels, _ = get_gibson_and_deep_door_2_dataset(True)

print(f'Train set size: {len(train)}', f'Validation set size: {len(validation)}')



