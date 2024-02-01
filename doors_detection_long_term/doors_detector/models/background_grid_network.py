import copy
import os
import time
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import torch
import torchvision.models
from torch import nn
from torch.hub import load_state_dict_from_url
from torch.nn import Identity, AdaptiveAvgPool2d
from torchvision.models.quantization import resnet50
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.ops import FeaturePyramidNetwork

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.generic_model import GenericModel, DESCRIPTION
from doors_detection_long_term.doors_detector.models.model_names import *
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

TEST: DESCRIPTION = 0
IMAGE_GRID_NETWORK: DESCRIPTION = 1
IMAGE_GRID_NETWORK_GIBSON_DD2: DESCRIPTION = 2
IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL: DESCRIPTION = 3
IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL_FLOOR1: DESCRIPTION = 4
IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL_FLOOR4: DESCRIPTION = 5
IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL_CHEMISTRY_FLOOR0: DESCRIPTION = 6
IMAGE_GRID_NETWORK_GIBSON_DD2_SMALL_HOUSE_MATTEO: DESCRIPTION = 7


class ResNet18FPN(ResNet):
    def __init__(self):
        super(ResNet18FPN, self).__init__(BasicBlock, [2, 2, 2, 2])

        #state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-5c106cde.pth', progress=True)
        #self.load_state_dict(state_dict)
        self.layer0 = copy.deepcopy(self.layer1)
        self.layer1[0].conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.layer1[0].downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.layer4 = nn.Identity()
        self.avgpool = nn.Identity()
        self.fc = nn.Identity()

    def _forward_impl(self, x: torch.Tensor) -> tuple:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer0(x)

        x1 = self.layer1(x)

        x2 = self.layer2(x1)

        x3 = self.layer3(x2)

        #x4 = self.layer4(x3)

        return x1, x2, x3


class MultipleConvolutions(nn.Module):
    def __init__(self, original_size: int, start_size: int, end_size: int):
        super(MultipleConvolutions, self).__init__()

        if start_size == original_size:
            modules = []
        else:
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


class FPNBackbone(nn.Module):
    def __init__(self, start_sizes: List[int], end_size: int,  image_grid_dimensions: List[Tuple[int, int]]):
        super(FPNBackbone, self).__init__()
        self.backbone = nn.Sequential(*[nn.Sequential() for _ in range(len(image_grid_dimensions))])
        self.upsample = nn.Sequential()

        for size in image_grid_dimensions[1:]:
            self.upsample.append(nn.Upsample(size=size, mode='nearest'))

        start_size = start_sizes[0]
        while start_size > end_size:
            for max_size, sequential in zip(start_sizes, self.backbone):
                if start_size <= max_size:
                    sequential.append(
                        MultipleConvolutions(original_size=start_size, start_size=start_size, end_size=start_size // 2),
                    )
                else:
                    sequential.append(nn.Identity())
            start_size //= 2

    def forward(self, features):
        for backbones in zip(*self.backbone):

            # Sum feature maps
            for feature_count in range(len(features) - 1):
                if not isinstance(backbones[feature_count + 1], nn.Identity):
                    features[feature_count + 1] = features[feature_count + 1] + self.upsample[feature_count](features[feature_count])

            for layer_count, layer in enumerate(backbones):
                features[layer_count] = layer(features[layer_count])

        return features

class ImageGridNetwork(GenericModel):
    def __init__(self, fpn_channels: int, image_grid_dimensions: List[Tuple[int, int]], model_name: ModelName, pretrained: bool, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(ImageGridNetwork, self).__init__(model_name, dataset_name, description)
        self._image_grid_dimensions = image_grid_dimensions
        self.fpn = ResNet18FPN()
        self.fpn_backbone = FPNBackbone(start_sizes=[256, 128, 64], end_size=16, image_grid_dimensions=image_grid_dimensions[::-1])

        self.adaptive_max_pooling_x0 = nn.AdaptiveMaxPool2d(output_size=image_grid_dimensions[0])
        self.adaptive_max_pooling_x1 = nn.AdaptiveMaxPool2d(output_size=image_grid_dimensions[1])
        self.adaptive_max_pooling_x2 = nn.AdaptiveMaxPool2d(output_size=image_grid_dimensions[2])

        self.upsample_x0 = nn.Upsample(size=image_grid_dimensions[0], mode='nearest')
        self.upsample_x1 = nn.Upsample(size=image_grid_dimensions[1], mode='nearest')
        self.upsample_x2 = nn.Upsample(size=image_grid_dimensions[2], mode='nearest')

        self.final_convolution = nn.Sequential(
            MultipleConvolutions(original_size=48, start_size=32, end_size=8),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        if pretrained:
            path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, images):
        x0, x1, x2 = self.fpn(images)
        x0 = self.adaptive_max_pooling_x0(x0)
        x1 = self.adaptive_max_pooling_x1(x1)
        x2 = self.adaptive_max_pooling_x2(x2)

        x2, x1, x0 = self.fpn_backbone([x2, x1, x0])
        x0 = self.upsample_x0(x0)
        x1 = self.upsample_x0(x1)
        x2 = self.upsample_x0(x2)

        x = torch.cat([x0, x1, x2], dim=1)
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
            loss_background.append(torch.nan_to_num(-torch.log(1-torch.mean(prediction[~image_grid.bool()]))))

        loss_background = torch.stack(loss_background)
        loss_target = torch.stack(loss_target)

        loss = torch.mean(loss_background, dim=0) + torch.mean(loss_target, dim=0)
        return loss
"""
image = torch.rand(1, 3, 240, 320)

bbox_model = ImageGridNetwork(fpn_channels=256, image_grid_dimensions=[(2**i, 2**i) for i in range(3, 6)][::-1], n_labels=3, model_name=IMAGE_GRID_NETWORK, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=IMAGE_GRID_NETWORK)

print(f'I PARAMTETRI SONO: {sum([np.prod(p.size()) for p in bbox_model.parameters()])}')
fpn_backbone = FPNBackbone(start_sizes=[256, 128, 64], end_size=16, image_grid_dimensions=[(2**i, 2**i) for i in range(3, 6)])
print(f'I PARAMTETRI SONO: {sum([np.prod(p.size()) for p in fpn_backbone.parameters()])}')
total = 0

for i in range(100):
    t = time.time()
    bbox_model(image)
    total+= time.time() - t
print(1/(total/100))
"""






