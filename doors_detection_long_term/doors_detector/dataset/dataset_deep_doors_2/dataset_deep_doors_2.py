import os

import cv2
import numpy as np
from gibson_env_utilities.doors_dataset.door_sample import DoorSample

from doors_detector.dataset.torch_dataset import TorchDataset


class DatasetDeepDoors2(TorchDataset):

    def load_sample(self, idx) -> DoorSample:
        row = self._dataframe.iloc[idx]
        file_name = row.file_name
        label = row.label
        depth_image_path = row.depth_image_path

        door_sample = DoorSample(label=1)

        door_sample.set_bgr_image(
            cv2.imread(
                os.path.join(self._dataset_path, 'door_detection', 'Images', file_name)
            )
        )
        door_sample.set_pretty_semantic_image(
            cv2.imread(
                os.path.join(self._dataset_path, 'door_detection', 'Annotations', file_name),

            )
        )
        door_sample.set_depth_image(
            cv2.imread(
                os.path.join(depth_image_path, file_name),
                cv2.IMREAD_GRAYSCALE
            )
        )

        bboxes = door_sample.get_bboxes_from_semantic_image()

        door_sample.set_bounding_boxes(np.array([(label, *rect) for rect in bboxes]))

        return door_sample