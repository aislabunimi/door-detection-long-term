import numpy as np
from generic_dataset.sample_generator import SampleGenerator
from generic_dataset.utilities.save_load_methods import save_cv2_image_bgr, load_cv2_image_bgr, \
    load_compressed_numpy_array, save_compressed_numpy_array, save_float, load_float

BoxFilteringExample = SampleGenerator(name='BoxFilteringExample', label_set={0, 1}) \
    .add_dataset_field(field_name='bgr_image', field_type=np.ndarray, save_function=save_cv2_image_bgr, load_function=load_cv2_image_bgr) \
    .add_dataset_field(field_name='detected_bounding_boxes', field_type=np.ndarray, default_value=np.array([]), load_function=load_compressed_numpy_array, save_function=save_compressed_numpy_array) \
    .add_dataset_field(field_name='gt_bounding_boxes', field_type=np.ndarray, default_value=np.array([]), load_function=load_compressed_numpy_array, save_function=save_compressed_numpy_array) \
    .add_dataset_field(field_name='example_type', field_type=float, save_function=save_float, load_function=load_float) \
.generate_sample_class()