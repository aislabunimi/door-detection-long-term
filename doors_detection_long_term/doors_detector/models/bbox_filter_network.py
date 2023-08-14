import os
from collections import OrderedDict
from typing import Tuple

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
IMAGE_TEST: DESCRIPTION = 1
TEST_IMAGE_GLOBAL_NETWORK: DESCRIPTION = 2
TEST_IMAGE_LOCAL_NETWORK: DESCRIPTION = 3
TEST_IMAGE_LOCAL_NETWORK_FINE_TUNE: DESCRIPTION = 4
TEST_IMAGE_LOCAL_NETWORK_SMALL: DESCRIPTION = 5
IMAGE_GRID_NETWORK: DESCRIPTION = 6


class SharedMLP(nn.Module):
    def __init__(self, channels, last_activation=nn.ReLU()):
        super(SharedMLP, self).__init__()
        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                       #nn.Dropout1d(p=0.8),
                       nn.BatchNorm1d(num_features=out_channels),
                       nn.ReLU()]
        layers[-1] = last_activation
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.shared_mlp(input)


class ResNet50FPN(ResNet):
    def __init__(self, channels=256):
        super(ResNet50FPN, self).__init__(Bottleneck, [3, 4, 6, 3])

        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)
        self.load_state_dict(state_dict)

        self.fpn = FeaturePyramidNetwork(in_channels_list=[64, 256, 512, 1024, 2048], out_channels=channels)

    def _forward_impl(self, x: torch.Tensor) -> list:
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

        pyramid_features = self.fpn(ordered_dict)

        return pyramid_features


class ImageGridNetwork(GenericModel):
    def __init__(self, fpn_channels: int, image_grid_dimensions: Tuple[int, int], model_name: ModelName, pretrained: bool, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(ImageGridNetwork, self).__init__(model_name, dataset_name, description)

        self.fpn = ResNet50FPN(channels=fpn_channels)
        self.batch_norm_after_fpn = nn.BatchNorm2d(fpn_channels)
        self.adaptive_max_pool_2d = nn.AdaptiveMaxPool2d(image_grid_dimensions)

        self.image_region = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=fpn_channels * image_grid_dimensions[0] * image_grid_dimensions[1], out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=image_grid_dimensions[0] * image_grid_dimensions[1]),
            nn.BatchNorm1d(image_grid_dimensions[0] * image_grid_dimensions[1]),
            nn.Unflatten(1, image_grid_dimensions),
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
        x = self.adaptive_max_pool_2d(x)
        image_region_features = self.image_region(x)
        image_region_features = image_region_features.transpose(1, 2)

        return image_region_features


class ImageGridNetworkLoss(nn.Module):
    def forward(self, predictions, image_grids):
        loss = torch.sum(torch.mean(-(torch.log(predictions) * image_grids + torch.log(1-image_grids) * (1-predictions)), dim=(1, 2)))
        return loss



