from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']

train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=2, folder_name='house1', train_size=0.75, use_negatives=False, all_test_set=True)

