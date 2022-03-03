import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from generic_dataset.dataset_manager import DatasetManager
from sklearn.utils import shuffle

from doors_detector.dataset.dataset_doors_final.door_sample import DoorSample
from doors_detector.dataset.dataset_doors_final.final_doors_dataset import DatasetDoorsFinal
from doors_detector.dataset.torch_dataset import TorchDataset, TRAIN_SET, SET, TEST_SET


class TorchDatasetModelOutput(DatasetDoorsFinal):
    def __init__(self, dataset_path: str,
                 dataframe: pd.DataFrame,
                 targets: Dict,
                 set_type: SET,
                 std_size: int,
                 max_size: int,
                 scales: List[int]):


        super(TorchDatasetModelOutput, self).__init__(
            dataset_path=dataset_path,
            dataframe=dataframe,
            set_type=set_type,
            std_size=std_size,
            max_size=max_size,
            scales=scales)
        self._targets = targets


    def __getitem__(self, idx):
        method = self.get_dataframe().method.iloc[idx]

        if method == 0:
            return super().__getitem__(idx)

        door_sample, folder_name, absolute_count = super().load_sample(idx)
        target = {}
        (h, w, _) = door_sample.get_bgr_image().shape
        target['size'] = torch.tensor([int(h), int(w)], dtype=torch.int)
        target['boxes'] = torch.tensor(self._targets[absolute_count]['bboxes'], dtype=torch.float)
        target['labels'] = torch.tensor(self._targets[absolute_count]['labels'], dtype=torch.long)
        target['folder_name'] = folder_name
        target['absolute_count'] = absolute_count

        # The BGR image is convert in RGB
        img, target = self._transform(Image.fromarray(door_sample.get_bgr_image()[..., [2, 1, 0]]), target)

        return img, target, door_sample


class DatasetCreatorFineTuneModelOutput:
    def __init__(self, dataset_path: str, folder_name: str, test_dataframe: pd.DataFrame):
        self._dataset_path = dataset_path
        self._folder_name = folder_name
        self._test_dataframe = test_dataframe
        self._absolute_counts: list = []
        self._targets = {}

    def add_train_sample(self, absolute_count: int, targets):
        self._absolute_counts.append(absolute_count)
        self._targets[absolute_count] = targets

    def create_datasets(self, random_state: int = 42):
        complete_dataframe = DatasetManager(dataset_path=self._dataset_path, sample_class=DoorSample).get_dataframe()
        folders = complete_dataframe.folder_name.unique().tolist()
        folders.remove(self._folder_name)
        folders = random.sample(folders, 2)
        complete_dataframe = complete_dataframe[complete_dataframe.folder_name.isin(folders) & (complete_dataframe.label == 1)]
        complete_dataframe = complete_dataframe.loc[random.sample(complete_dataframe.index.tolist(), 0)]

        dataframe_train = pd.DataFrame(data={
            'folder_name': [self._folder_name for _ in range(len(self._absolute_counts))] + complete_dataframe.folder_name.tolist(),
            'folder_absolute_count': self._absolute_counts + complete_dataframe.folder_absolute_count.tolist(),
            'method': [1 for _ in range(len(self._absolute_counts))] + [0 for _ in range(len(complete_dataframe.index))]
        })
        dataframe_train = shuffle(dataframe_train, random_state=random_state)
        dataframe_test = self._test_dataframe[self._test_dataframe.label == 1]
        return (TorchDatasetModelOutput(dataset_path=self._dataset_path, dataframe=dataframe_train, targets=self._targets, set_type=TRAIN_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]),
                DatasetDoorsFinal(self._dataset_path, dataframe_test, TEST_SET, std_size=256, max_size=800, scales=[256 + i * 32 for i in range(11)]))


