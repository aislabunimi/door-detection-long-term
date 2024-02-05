from enum import Enum

import cv2
import numpy as np
import torch
from generic_dataset.dataset_folder_manager import DatasetFolderManager
from generic_dataset.dataset_manager import DatasetManager
from src.bounding_box import BoundingBox
from src.utils.enumerators import BBType, BBFormat, CoordinatesType
from sklearn.utils import shuffle

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.box_filtering_example import BoxFilteringExample
from doors_detection_long_term.doors_detector.dataset.torch_dataset import TorchDatasetBBoxes, TRAIN_SET, TEST_SET
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import xywh2xyxy
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import box_filtering_dataset_path


class ExampleType(Enum):
    TRAINING = 1
    TEST = 2

class DatasetCreatorBBoxes:
    def __init__(self, max_bboxes: int = 200):
        self._colors = {0: (0, 0, 255), 1: (0, 255, 0)}
        self._max_bboxes = max_bboxes
        self._img_count = 0
        self._training_bboxes = {
            'images': [],
            'gt_bboxes': [],
            'bboxes': [],
            'bboxes_matched': [],
            'gt_bboxes_matched': []
        }
        self._test_bboxes = {
            'images': [],
            'gt_bboxes': [],
            'bboxes': [],
            'bboxes_matched': [],
            'gt_bboxes_matched': []
        }

        self._last_saved_example_training = 0
        self._last_saved_example_test = 0

        self._folder_name = ''

    def visualize_bboxes(self, show_filtered: bool = False, bboxes_type: ExampleType = ExampleType.TRAINING):
        if bboxes_type == ExampleType.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == ExampleType.TRAINING:
            bboxes_dict = self._training_bboxes
        for i, (image, bboxes, target_bboxes, matched_bboxes) in enumerate(zip(bboxes_dict['images'], bboxes_dict['bboxes'], bboxes_dict['gt_bboxes'], bboxes_dict['bboxes_matched'])):
            img_size = image.shape
            show_image = image.copy()
            target_image = image.copy()
            background_image = image.copy()
            matched_images = []

            for bbox in target_bboxes:
                x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
                target_image = cv2.rectangle(target_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(float(bbox.get_class_id()))], 2)

                # Create a matched image for each gt bbox
                matched_image = image.copy()
                matched_image = cv2.rectangle(matched_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

                for detected_box, gt_box in matched_bboxes:
                    if gt_box == bbox:
                        x1, y1, x2, y2 = detected_box.get_absolute_bounding_box(BBFormat.XYX2Y2)
                        matched_image = cv2.rectangle(matched_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(float(bbox.get_class_id()))], 2)
                matched_images.append(matched_image)

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox.get_absolute_bounding_box(BBFormat.XYX2Y2)
                show_image = cv2.rectangle(show_image, (int(x1), int(y1)), (int(x2), int(y2)), self._colors[int(float(bbox.get_class_id()))], 2)

            for detected_box, gt_box in matched_bboxes:
                if gt_box == None:
                    x1, y1, x2, y2 = detected_box.get_absolute_bounding_box(BBFormat.XYX2Y2)
                    background_image = cv2.rectangle(background_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)

            show_image = cv2.hconcat([target_image, show_image] + matched_images + [background_image])
            cv2.imshow('show', show_image)
            cv2.waitKey()

    def _match_detected_bboxes(self, dataset_dict, iou_threshold_matching, print_stats=False):
        gt_matching_images = []
        matching_images = []
        for detected_bboxes_image, gt_bboxes_image in zip(dataset_dict['bboxes'], dataset_dict['gt_bboxes']):
            gts = []
            gt_matching=[]
            matching = []
            for detected_bbox in detected_bboxes_image:
                matched_gt = None
                match_iou = -1
                for gt_bbox in gt_bboxes_image:
                    iou = BoundingBox.iou(detected_bbox, gt_bbox)
                    if iou >= match_iou and iou >= iou_threshold_matching:
                        matched_gt = gt_bbox
                        match_iou = iou
                matching.append((detected_bbox, matched_gt))
                if matched_gt is not None:
                    if matched_gt not in gts:
                        gts.append(matched_gt)
                        gt_matching.append((matched_gt, []))
                    for gt, l in gt_matching:
                        if gt == matched_gt:
                            l.append(detected_bbox)
                            break
            matching_images.append(matching)
            gt_matching_images.append(gt_matching)
        dataset_dict['bboxes_matched'] = matching_images
        dataset_dict['gt_bboxes_matched'] = gt_matching_images
        if print_stats:
            confs = []
            lens = []
            poorly_matched_gt = []
            for i in gt_matching_images:

                for gt_box, matched_bboxes in i:
                    lens.append(len(matched_bboxes))
                    confs.append(sum([b.get_confidence() for b in matched_bboxes]) / len(matched_bboxes))
                    if len(matched_bboxes) < 3:
                        poorly_matched_gt.append(1)
            if len(lens) > 0:
                print(f'Il numero medio di match per GT = {sum(lens) / len(lens)}')
                print(f'Il confidence media dei detected matched = {sum(confs) / len(confs)}')
                print(f'I gt matchati con pochi bboxes è = {sum(poorly_matched_gt)}')

    def select_n_bounding_boxes(self, num_bboxes: int):
        for i, bboxes in enumerate(self._training_bboxes['bboxes']):
            self._training_bboxes['bboxes'][i] = sorted(bboxes, key=lambda x: x.get_confidence(), reverse=True)[:num_bboxes]

        for i, bboxes in enumerate(self._test_bboxes['bboxes']):
            self._test_bboxes['bboxes'][i] = sorted(bboxes, key=lambda x: x.get_confidence(), reverse=True)[:num_bboxes]

    def match_bboxes_with_gt(self, iou_threshold_matching: float = 0.5, print_stats=False):
        self._match_detected_bboxes(self._training_bboxes, iou_threshold_matching, print_stats=print_stats)
        self._match_detected_bboxes(self._test_bboxes, iou_threshold_matching, print_stats=print_stats)

    def create_datasets(self, shuffle_boxes: bool = False, apply_transforms_to_train: bool = False, random_state: int = 42):
        return (TorchDatasetBBoxes(bboxes_dict=self._training_bboxes, set_type=TRAIN_SET if apply_transforms_to_train else TEST_SET, shuffle=shuffle_boxes),
                TorchDatasetBBoxes(bboxes_dict=self._test_bboxes, set_type=TEST_SET, shuffle=False))

    def set_folder_name(self, folder_name: str):
        self._folder_name = folder_name

    def export_dataset(self):
        if type(self._folder_name) != str or self._folder_name == '':
            raise Exception('Please, before save dataset set folder name')

        folder_manager = DatasetFolderManager(dataset_path=box_filtering_dataset_path, folder_name=self._folder_name, sample_class=BoxFilteringExample)

        def save_dictionary(dataset, example_type):
            for image, gt_boxes, bboxes in zip(dataset['images'], dataset['gt_bboxes'], dataset['bboxes']):
                example = BoxFilteringExample()

                example.set_label(0) if example_type == ExampleType.TRAINING else example.set_label(1)

                example.set_bgr_image((image*255).astype(np.uint8))
                #example.set_example_type(float(1 if example_type == ExampleType.TRAINING else 2))

                gt_bboxes_converted = [list(box.get_absolute_bounding_box(BBFormat.XYX2Y2)) + [int(box.get_class_id()), 1.0] for box in gt_boxes]
                bboxes_converted = [list(box.get_absolute_bounding_box(BBFormat.XYX2Y2)) + [int(box.get_class_id()), box.get_confidence()] for box in bboxes]

                example.set_gt_bounding_boxes(np.array(gt_bboxes_converted))
                example.set_detected_bounding_boxes(np.array(bboxes_converted))
                folder_manager.save_sample(example, use_thread=True)

            # Remove saved examples
            dataset['images'] = []
            dataset['gt_bboxes'] = []
            dataset['bboxes'] = []

        save_dictionary(self._training_bboxes, ExampleType.TRAINING)

        save_dictionary(self._test_bboxes, ExampleType.TEST)

        folder_manager.save_metadata()

    def load_dataset(self, folder_name: str):
        dataset_manager = DatasetManager(dataset_path=box_filtering_dataset_path, sample_class=BoxFilteringExample)
        if folder_name not in dataset_manager.get_folder_names():
            raise Exception('Folder name doen\'t exists')

        folder_manager = DatasetFolderManager(dataset_path=box_filtering_dataset_path, folder_name=folder_name, sample_class=BoxFilteringExample)
        for (label, relative_count) in folder_manager.get_samples_information():
            sample = folder_manager.load_sample_using_relative_count(label=label, relative_count=relative_count, use_thread=False)
            dictionary = self._training_bboxes if ExampleType(int(sample.get_example_type())) == ExampleType.TRAINING else self._test_bboxes
            dictionary['images'].append(sample.get_bgr_image())
            gt_bboxes = []
            for x1, y1, x2, y2, label, conf in sample.get_gt_bounding_boxes():
                gt_bboxes.append(BoundingBox(
                    image_name=str(relative_count),
                    class_id=str(int(float(label))),
                    coordinates=(int(x1), int(y1), int(x2), int(y2)),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYX2Y2,
                ))
            dictionary['gt_bboxes'].append(gt_bboxes)

            bboxes = []
            for x1, y1, x2, y2, label, conf in sample.get_detected_bounding_boxes():
                bboxes.append(BoundingBox(
                    image_name=str(relative_count),
                    class_id=str(int(float(label))),
                    coordinates=(int(x1), int(y1), int(x2), int(y2)),
                    bb_type=BBType.DETECTED,
                    format=BBFormat.XYX2Y2,
                    confidence=conf
                ))
            dictionary['bboxes'].append(bboxes)

    def add_yolo_bboxes(self, images, targets, preds, bboxes_type: ExampleType):
        if bboxes_type == ExampleType.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == ExampleType.TRAINING:
            bboxes_dict = self._training_bboxes

        img_size = images.size()[2:][::-1]
        for image in images:
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            bboxes_dict['images'].append(image)


        img_count_temp = self._img_count
        for target in targets:
            gt_boxes = []
            for label, [x, y, w, h] in zip(target['labels'].tolist(), target['boxes'].tolist()):
                gt_boxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x, y, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                    img_size=img_size,
                    type_coordinates=CoordinatesType.RELATIVE,
                ))
            bboxes_dict['gt_bboxes'].append(gt_boxes)
            self._img_count += 1

        for bboxes in preds:
            coords = xywh2xyxy(bboxes[:, :4])
            conf = bboxes[:, 5:] * bboxes[:, 4:5]
            conf, labels = conf.max(1, keepdim=True)
            conf = torch.squeeze(conf)
            labels = torch.squeeze(labels)
            detected_boxes = []
            for (x1, y1, x2, y2), score, label in zip(coords.tolist(), conf.tolist(), labels.tolist()):
                label = int(label)
                if label >= 0:
                    box = BoundingBox(
                        image_name=str(img_count_temp),
                        class_id=str(label),
                        type_coordinates=CoordinatesType.ABSOLUTE,
                        coordinates=(x1, y1, x2 - x1, y2 - y1),
                        bb_type=BBType.DETECTED,
                        format=BBFormat.XYWH,
                        confidence=score,
                        img_size=img_size
                    )

                    x1, y1, x2, y2 = box.get_absolute_bounding_box(BBFormat.XYX2Y2)
                    if x2 > x1 + 1 and y2 > y1 + 1:
                        detected_boxes.append(box)

            detected_boxes = sorted(detected_boxes, key=lambda x: x.get_confidence(), reverse=True)[:self._max_bboxes]
            bboxes_dict['bboxes'].append(detected_boxes)
            img_count_temp += 1

    def add_faster_rcnn_bboxes(self, images, targets, preds, bboxes_type: ExampleType):
        if bboxes_type == ExampleType.TEST:
            bboxes_dict = self._test_bboxes
        elif bboxes_type == ExampleType.TRAINING:
            bboxes_dict = self._training_bboxes

        img_size = images.size()[2:][::-1]
        for image in images:
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            bboxes_dict['images'].append(image)


        img_count_temp = self._img_count
        for target in targets:
            gt_boxes = []
            for label, [x, y, w, h] in zip(target['labels'].tolist(), target['boxes'].tolist()):
                gt_boxes.append(BoundingBox(
                    image_name=str(self._img_count),
                    class_id=str(label),
                    coordinates=(x, y, w, h),
                    bb_type=BBType.GROUND_TRUTH,
                    format=BBFormat.XYWH,
                    img_size=img_size,
                    type_coordinates=CoordinatesType.RELATIVE,
                ))
            bboxes_dict['gt_bboxes'].append(gt_boxes)
            self._img_count += 1

        for prediction in preds:
            detected_boxes = []

            for [x1, y1, x2, y2], label, score in zip(prediction['boxes'].tolist(), prediction['labels'].tolist(), prediction['scores'].tolist()):

                if label >= 0:
                    box = BoundingBox(
                            image_name=str(img_count_temp),
                            class_id=str(label),
                            coordinates=(x1, y1, x2 - x1, y2 - y1),
                            bb_type=BBType.DETECTED,
                            format=BBFormat.XYWH,
                            confidence=score
                        )

                    x1, y1, x2, y2 = box.get_absolute_bounding_box(BBFormat.XYX2Y2)
                    if x2 > x1 + 1 and y2 > y1 + 1:
                        detected_boxes.append(box)
            img_count_temp += 1
            detected_boxes = sorted(detected_boxes, key=lambda x: x.get_confidence(), reverse=True)[:self._max_bboxes]
            bboxes_dict['bboxes'].append(detected_boxes)