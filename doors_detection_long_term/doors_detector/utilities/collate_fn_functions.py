import os
import random
from typing import Tuple

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def collate_fn(batch_data):
    # Batch data is a list of n tuple, where tuple[0] is the img while tuple[1] are targets (labels, bounding boxes ecc)
    # Batch data is transformed in a list where list[0] contains a list of the images and list[1] contains a list of targets
    batch_data = list(zip(*batch_data))

    def _max_by_axis(the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    sizes = [list(img.shape) for img in batch_data[0]]

    max_sizes = _max_by_axis(sizes)

    # Replace batch_data[0] with a tensor containing all batch images

    final_size = [len(batch_data[0])] + max_sizes
    b, c, h, w = final_size
    device = batch_data[0][0].device
    dtype = batch_data[0][0].dtype

    tensor = torch.zeros(final_size, dtype=dtype, device=device)

    for img, pad_img in zip(batch_data[0], tensor):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    batch_data[0] = tensor

    return tuple(batch_data[:2])

def collate_fn_yolov5(batch):
    images, targets = collate_fn(batch)

    batch_size_width, batch_size_height = images.size()[2], images.size()[3]
    final_size = [i for i in images.size()[:2]]

    translate_w, translate_h = 0., 0.
    # If the images have different width and height, make them squared adding padding
    modify = False
    if batch_size_width > batch_size_height:
        final_size += [batch_size_width, batch_size_width]
        modify = True
    else:
        final_size += [batch_size_height, batch_size_height]
        modify = True

    if final_size[2] % 32 != 0 or final_size[3] % 32 != 0:
        final_size[2] = final_size[3] = final_size[2] + 32 - final_size[2] % 32

    batch_size_width, batch_size_height = final_size[2], final_size[3]
    final_size = tuple(final_size)

    if modify:
        tensor = torch.zeros(size=final_size, dtype=images.dtype, device=images.device)

        translate_w = (tensor.size()[3] - images.size()[3]) / tensor.size()[3] / 2
        translate_h =  (tensor.size()[2] - images.size()[2]) / tensor.size()[2] / 2

        for img, pad_img in zip(images, tensor):
            pad_img[: img.size()[0], int((final_size[2] - img.size()[1]) / 2.) : img.shape[1] + int((final_size[2] - img.size()[1]) / 2.), int((final_size[3] - img.size()[2]) / 2.) : img.shape[2] + int((final_size[3] - img.size()[2]) / 2.)].copy_(img)
        images = tensor


    converted_boxes = []
    for i, target in enumerate(targets):
        real_size_width, real_size_height = target['size'][1], target['size'][0]
        scale_boxes = torch.tensor([[real_size_width / batch_size_width, real_size_height / batch_size_height,
                                     real_size_width / batch_size_width, real_size_height / batch_size_height]])

        converted_boxes.append(torch.cat([
            torch.tensor([[i] for _ in range(int(list(target['labels'].size())[0]))]),
            torch.reshape(target['labels'], (target['labels'].size()[0], 1)),
            target['boxes'] * scale_boxes + torch.tensor([[translate_w, translate_h, 0., 0.]])
        ], dim=1))

        # Update targets to new images
        target['boxes'] = target['boxes'] * scale_boxes + torch.tensor([[translate_w, translate_h, 0., 0.]])

    converted_boxes = torch.cat(converted_boxes, dim=0)
    return images, targets, converted_boxes

def collate_fn_faster_rcnn(batch):
    images, targets, converted_boxes = collate_fn_yolov5(batch)
    batch_size_width, batch_size_height = images.size()[2], images.size()[3]
    new_targets = []
    for i, image in enumerate(images):
        t = {}
        current_boxes = converted_boxes[converted_boxes[:, 0].to(torch.int) == i]
        t['labels'] = current_boxes[:, 1].to(torch.int) + 1
        t['boxes'] = torch.cat((current_boxes[:, 2:4] - (current_boxes[:, 4:] / 2), current_boxes[:, 2:4] + (current_boxes[:, 4:] / 2)), 1) * torch.tensor([[batch_size_width, batch_size_height, batch_size_width, batch_size_width]])
        t['area'] = torch.prod(t['boxes'][:, 2:] - t['boxes'][:, :2], 1)
        t['iscrowd'] = torch.zeros(t['area'].size()[0]).to(torch.int)
        new_targets.append(t)

    return images, targets, new_targets

def collate_fn_bboxes(image_grid_dimensions: Tuple[int, int] = (20, 20), use_confidence: bool = True):
    def _collate_fn_bboxes(batch_data):
        """
        The bounding boxes come encoded as [cx, cy, w, h]
        Targets is a list of dictionaries, each of them contains the boxes, confidences and labels (encoded in 0,1 vector) of the relative image
        Converted_bbxes contains a list of lists of bboxes encoded as [cx, cy, w, h]
        :param batch_data:
        :return:
        images,
        detected_bounding_boxes encoded as [cx, cy, w, h, original_confidence, original_label_encoded],
        fixed_bounding_boxes,
        confidences 0 if the bbox is suppressed (background), 1 otherwise
        labels_encoded: the ground truth labels of the bounding boxes [background, closed, open]
        ious: the iou with the target bbox. It is 0 if the bbox is background
        target_boxes: the target bounding boxes encoded as [cx, cy, w, h, label]
        image_grid: the grid of the image
        detected_boxes_grid: the coordinates of the detected boxes in the image grid, encoded as [x1, y1, x2, y2]

        """
        images, targets = collate_fn(batch_data)

        batch_size_width, batch_size_height = images.size()[3], images.size()[2]

        target_boxes = []
        fixed_boxes = []
        detected_boxes = []
        confidences = []
        labels_encoded = []
        ious = []
        image_grids = []
        detected_boxes_grid = []
        for i, target in enumerate(targets):

            # Rescale bboxes according to the batch global size
            real_size_width, real_size_height = target['size'][1], target['size'][0]
            scale_boxes = torch.tensor([[real_size_width / batch_size_width, real_size_height / batch_size_height,
                                         real_size_width / batch_size_width, real_size_height / batch_size_height]])
            target['fixed_boxes'] = target['fixed_boxes'] * scale_boxes
            target['detected_boxes'] = target['detected_boxes'] * scale_boxes
            target['target_boxes'][:, :4] *= scale_boxes

            fixed_boxes.append(target['fixed_boxes'])
            target_boxes.append(target['target_boxes'])

            if use_confidence:
                detected_boxes.append(
                    torch.cat([target['detected_boxes'],
                               target['original_confidences'],
                               target['original_labels']], dim=1)
                )
            else:
                detected_boxes.append(
                    torch.cat([target['detected_boxes'],
                               target['original_labels']], dim=1)
                )

            # Grid computation
            step_w = (real_size_width / image_grid_dimensions[0]).item()
            step_h = (real_size_height / image_grid_dimensions[1]).item()
            grid = torch.zeros(image_grid_dimensions)
            targets_x1y1x2y2 = torch.cat([target['target_boxes'][:, 0:1] - target['target_boxes'][:, 2:3] / 2,
                                          target['target_boxes'][:, 1:2] - target['target_boxes'][:, 3:4] / 2,
                                          target['target_boxes'][:, 0:1] + target['target_boxes'][:, 2:3] / 2,
                                          target['target_boxes'][:, 1:2] + target['target_boxes'][:, 3:4] / 2], dim=1)
            targets_x1y1x2y2 = targets_x1y1x2y2 * torch.tensor([real_size_width, real_size_height, real_size_width, real_size_height])
            # print(targets_x1y1x2y2)
            targets_x1y1x2y2 = torch.cat([targets_x1y1x2y2, target['target_boxes'][:, 4: 5]], dim=1)
            for x1, y1, x2, y2, label in targets_x1y1x2y2.tolist():
                #      print('LABEL', label)
                mapped_x1 = round(x1 / step_w)
                mapped_x2 = round(x2 / step_w)
                mapped_y1 = round(y1 / step_h)
                mapped_y2 = round(y2 / step_h)
                if mapped_x1 == mapped_x2:
                    mapped_x2 += 1
                if mapped_y1 == mapped_y2:
                    mapped_y2 += 1
                #       print(mapped_x1, mapped_x2, mapped_y1, mapped_y2)
                #print('label encoded', torch.tensor([0 if i != label + 1 else 1 for i in range(3)], device = targets_batch.device))
                grid[mapped_x1: mapped_x2, mapped_y1:mapped_y2] = 1

            # COnverted detected bboxes in grid
            detected_x1y1x2y2 = torch.cat([target['detected_boxes'][:, 0:1] - target['detected_boxes'][:, 2:3] / 2,
                                          target['detected_boxes'][:, 1:2] - target['detected_boxes'][:, 3:4] / 2,
                                          target['detected_boxes'][:, 0:1] + target['detected_boxes'][:, 2:3] / 2,
                                          target['detected_boxes'][:, 1:2] + target['detected_boxes'][:, 3:4] / 2], dim=1)
            detected_x1y1x2y2 = detected_x1y1x2y2 * torch.tensor([real_size_width, real_size_height, real_size_width, real_size_height])
            detected_x1y1x2y2 = detected_x1y1x2y2 / torch.tensor([step_w, step_h, step_w, step_h])
            detected_x1y1x2y2 = torch.round(detected_x1y1x2y2).type(torch.int32)
            detected_boxes_grid.append(detected_x1y1x2y2)
            image_grids.append(grid)
            confidences.append(target['confidences'])
            labels_encoded.append(target['labels_encoded'])
            ious.append(target['ious'])

        fixed_bboxes = torch.stack(fixed_boxes, dim=0)
        detected_boxes = torch.stack(detected_boxes, dim=0)
        confidences = torch.stack(confidences, dim=0)
        labels_encoded = torch.stack(labels_encoded, dim=0)
        ious = torch.stack(ious, dim=0)
        image_grids = torch.stack(image_grids, dim=0)
        detected_boxes_grid = torch.stack(detected_boxes_grid, dim=0)

        return images, torch.transpose(detected_boxes, 1, 2), fixed_bboxes, confidences, labels_encoded, ious, target_boxes, image_grids, detected_boxes_grid
    return _collate_fn_bboxes


