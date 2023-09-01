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

        """nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=fpn_channels * image_grid_dimensions[0] * image_grid_dimensions[1], out_features=4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=image_grid_dimensions[0] * image_grid_dimensions[1]),
            nn.BatchNorm1d(image_grid_dimensions[0] * image_grid_dimensions[1]),
            nn.Unflatten(1, image_grid_dimensions),
            nn.Sigmoid(),
        )"""

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


class ImageGridNetworkLoss(nn.Module):
    def forward(self, predictions, image_grids, target_boxes_grid):
        """
        loss_boxes = []

        for batch, targets in enumerate(target_boxes_grid):
            mean_boxes_batch = []
            for x1, y1, x2, y2 in targets:
                mean_boxes_batch.append(predictions[batch, x1:x2, y1:y2].mean())
            mean_boxes_batch = torch.stack(mean_boxes_batch)
            mean_boxes_batch = torch.sum(-torch.log(mean_boxes_batch))

            loss_boxes.append(mean_boxes_batch)
        loss_boxes = torch.stack(loss_boxes)
        if torch.count_nonzero(torch.isnan(loss_boxes)) > 0:
            print('ERRORE')
        loss_background = []
        for pred, grid in zip(predictions, image_grids):
            image_background_loss = pred[grid == 0].mean()
            #print(image_background_loss, torch.log(1-image_background_loss.mean()), torch.square(torch.log(1-image_background_loss.mean())))
            image_background_loss = -torch.log(1-image_background_loss)
            loss_background.append(image_background_loss)

        loss_background = torch.nan_to_num(torch.stack(loss_background))
        if torch.count_nonzero(torch.isnan(loss_background)) > 0:
            print('ERRORE')
        loss = loss_boxes + loss_background
        loss = torch.sum(loss)

        #loss = torch.sum(torch.mean(-(torch.log(predictions) * image_grids + torch.log(1-predictions) * (1-image_grids)), dim=(1, 2)))
        """
        loss_background = []
        loss_target = []
        for prediction, image_grid in zip(predictions, image_grids[tuple(predictions.size()[1:])]):
            loss_target.append(-torch.log(torch.mean(prediction[image_grid.bool()])))
            loss_background.append(torch.nan_to_num(-torch.log(1-torch.mean(prediction[~image_grid.bool()]))))

        loss_background = torch.stack(loss_background)
        loss_target = torch.stack(loss_target)

        loss = torch.mean(loss_background, dim=0) + torch.mean(loss_target, dim=0)
        return loss

#image = torch.rand(4, 3, 240, 320)

#bbox_model = ImageGridNetwork(fpn_channels=256, image_grid_dimensions=(20,20), n_labels=3, model_name=IMAGE_GRID_NETWORK, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=IMAGE_GRID_NETWORK)
#i = bbox_model(image)
#print(i.size())



