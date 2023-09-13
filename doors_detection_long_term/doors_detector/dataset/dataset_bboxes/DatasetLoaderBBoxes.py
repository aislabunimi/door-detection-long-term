from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.dataset_manager import DatasetManager
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.box_filtering_example import BoxFilteringExample
from doors_detection_long_term.doors_detector.dataset.torch_dataset import TorchDatasetBBoxes, TRAIN_SET, TEST_SET
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import box_filtering_dataset_path


class DatasetLoaderBBoxes:
    def __init__(self, folder_name: str):
        self._dataset_manager = DatasetManager(dataset_path=box_filtering_dataset_path, sample_class=BoxFilteringExample)

        if folder_name not in self._dataset_manager.get_folder_names():
            raise Exception('Folder dataset does not exist!!')

        self._folder_name = folder_name
        self._dataframe = self._dataset_manager.get_dataframe()
        self._dataframe = self._dataframe[self._dataframe.folder_name == folder_name]

    def create_dataset(self,
                       max_bboxes: int,
                       iou_threshold_matching: float,
                       apply_transforms_to_train: bool = False,
                       shuffle_boxes: bool = False,
                       random_state: int = 42):

        shuffled_dataframe = shuffle(self._dataframe, random_state=random_state)

        train = shuffled_dataframe[(shuffled_dataframe.label == 0)]
        test = shuffled_dataframe[(shuffled_dataframe.label == 1)]

        return (TorchDatasetBBoxes(dataset_manager=self._dataset_manager, dataframe=train, folder_name=self._folder_name, set_type=TRAIN_SET if apply_transforms_to_train else TEST_SET, max_bboxes=max_bboxes, iou_threshold_matching=iou_threshold_matching, shuffle_boxes=shuffle_boxes),
                TorchDatasetBBoxes(dataset_manager=self._dataset_manager, dataframe=test, folder_name=self._folder_name, set_type=TEST_SET, max_bboxes=max_bboxes, iou_threshold_matching=iou_threshold_matching, shuffle_boxes=False))
