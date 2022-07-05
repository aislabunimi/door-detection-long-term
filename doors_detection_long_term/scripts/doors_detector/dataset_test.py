from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis, get_final_doors_dataset

house = 'house1'
train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=2, folder_name=house, train_size=0.75, use_negatives=True)
print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

train, test, labels, _ = get_final_doors_dataset(experiment=2, folder_name=house, train_size=0.75, use_negatives=True)
print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')
