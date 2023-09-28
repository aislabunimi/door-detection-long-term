import os
from collections import OrderedDict
from typing import Tuple, List

import torch
import torchvision.models
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import Identity, AdaptiveAvgPool2d
from torchvision.models.quantization import resnet50
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.ops import FeaturePyramidNetwork

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.generic_model import GenericModel, DESCRIPTION
from doors_detection_long_term.doors_detector.models.model_names import *
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

TEST: DESCRIPTION = 0
IMAGE_GRID_NETWORK: DESCRIPTION = 1


class ResNet50FPN(ResNet):
    def __init__(self, channels=256):
        super(ResNet50FPN, self).__init__(Bottleneck, [3, 4, 6, 3])

        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)
        self.load_state_dict(state_dict)

        self.fpn = FeaturePyramidNetwork(in_channels_list=[64, 256, 512, 1024, 2048], out_channels=channels)

    def _forward_impl(self, x: torch.Tensor) -> tuple:
        ordered_dict = OrderedDict()

        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)
        ordered_dict['x0'] = x0

        x1 = self.layer1(x)
        ordered_dict['x1'] = x1

        x2 = self.layer2(x1)
        ordered_dict['x2'] = x2

        x3 = self.layer3(x2)
        ordered_dict['x3'] = x3

        x4 = self.layer4(x3)
        ordered_dict['x4'] = x4

        #pyramid_features = self.fpn(ordered_dict)

        return ordered_dict

"""
class ImageGridNetwork(GenericModel):
    def __init__(self, fpn_channels: int, image_grid_dimensions: Tuple[int, int], model_name: ModelName, pretrained: bool, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(ImageGridNetwork, self).__init__(model_name, dataset_name, description)
        self._image_grid_dimensions = image_grid_dimensions
        self.fpn = ResNet50FPN(channels=fpn_channels)
        self.batch_norm_after_fpn = nn.BatchNorm2d(fpn_channels)

        if image_grid_dimensions is not None:
            self.adaptive_max_pool_2d = nn.AdaptiveMaxPool2d(image_grid_dimensions)

        self.image_region = nn.Sequential(
            nn.Conv2d(in_channels=fpn_channels, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        if pretrained:
            if pretrained:
                path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, images):
        x = self.fpn(images)['x2']
        if self._image_grid_dimensions is not None:
            x = self.adaptive_max_pool_2d(x)
        image_region_features = self.image_region(x).squeeze(1)
        image_region_features = image_region_features.transpose(1, 2)

        return image_region_features

"""

class MultipleConvolutions(nn.Module):
    def __init__(self, original_size: int, start_size: int, end_size: int):
        super(MultipleConvolutions, self).__init__()

        modules = [
            nn.Conv2d(in_channels=original_size, out_channels=start_size, kernel_size=1),
            nn.BatchNorm2d(num_features=start_size),
            nn.ReLU()
        ]

        while start_size > end_size:
            modules += [
                nn.Conv2d(in_channels=start_size, out_channels=start_size, kernel_size=5, padding=2),
                nn.BatchNorm2d(start_size),
                nn.Conv2d(in_channels=start_size, out_channels=start_size, kernel_size=3, padding=1),
                nn.BatchNorm2d(start_size),
                nn.Conv2d(in_channels=start_size, out_channels=start_size // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(start_size // 2),
                nn.ReLU(),
            ]
            start_size //= 2

        self.convolutions = nn.Sequential(*modules)

    def forward(self, x):
        return self.convolutions(x)


class ImageGridNetwork(GenericModel):
    def __init__(self, fpn_channels: int, image_grid_dimensions: List[Tuple[int, int]], model_name: ModelName, pretrained: bool, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(ImageGridNetwork, self).__init__(model_name, dataset_name, description)
        self._image_grid_dimensions = image_grid_dimensions
        self.fpn = ResNet50FPN(channels=fpn_channels)

        #self.conv_x1 = nn.Sequential(
         #   nn.AdaptiveMaxPool2d(output_size=image_grid_dimensions[0]),
          #  MultipleConvolutions(original_size=256, start_size=256, end_size=64)
        #)
        self.conv_x2 = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=image_grid_dimensions[1]),
            MultipleConvolutions(original_size=512, start_size=512, end_size=128),
            nn.Upsample(size=image_grid_dimensions[1], mode='nearest')
        )
        self.conv_x3 = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=image_grid_dimensions[2]),
            MultipleConvolutions(original_size=1024, start_size=512, end_size=64),
            nn.Upsample(size=image_grid_dimensions[1], mode='nearest')
        )
        self.conv_x4 = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=image_grid_dimensions[3]),
            MultipleConvolutions(original_size=2048, start_size=512, end_size=64),
            nn.Upsample(size=image_grid_dimensions[1], mode='nearest')
        )

        self.final_convolution = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        if pretrained:
            path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, images):
        multi_scales_maps = self.fpn(images)
        #x1 = self.conv_x1(multi_scales_maps['x1'])
        x2 = self.conv_x2(multi_scales_maps['x2'])
        x3 = self.conv_x3(multi_scales_maps['x3'])
        x4 = self.conv_x4(multi_scales_maps['x4'])

        x = torch.cat([x2, x3, x4], dim=1)
        image_region_features = self.final_convolution(x).squeeze(1)
        image_region_features = image_region_features.transpose(1, 2)

        return image_region_features

class ImageGridNetworkLoss(nn.Module):
    def forward(self, predictions, image_grids, target_boxes_grid):

        loss_background = []
        loss_target = []
        #print(tuple(predictions.size()[1:]))
        for prediction, image_grid in zip(predictions, image_grids[tuple(predictions.size()[1:])]):
            loss_target.append(torch.nan_to_num(-torch.log(torch.mean(prediction[image_grid.bool()]))))
            p = 0.3 if torch.count_nonzero(image_grid.bool()) == 0 else 1.0
            loss_background.append(p*torch.nan_to_num(-torch.log(1-torch.mean(prediction[~image_grid.bool()]))))


        loss_background = torch.stack(loss_background)
        loss_target = torch.stack(loss_target)

        loss = torch.mean(loss_background, dim=0) + torch.mean(loss_target, dim=0)
        return loss
"""
image = torch.rand(4, 3, 240, 320)

model = ResNet50FPN()
#x1, x2, x3, x4 = model(image)
#print(x1.size(), x2.size(), x3.size(), x4.size())

bbox_model = ImageGridNetwork(fpn_channels=256, image_grid_dimensions=[(2**i, 2**i) for i in range(3, 7)][::-1], n_labels=3, model_name=IMAGE_GRID_NETWORK, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=IMAGE_GRID_NETWORK)
print(bbox_model)
i = bbox_model(image)
print(i.size())

"""



