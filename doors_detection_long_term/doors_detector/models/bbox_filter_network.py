import os

import torch
import torchvision.models
from torch import nn
from torch.nn import Identity, AdaptiveAvgPool2d

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET
from doors_detection_long_term.doors_detector.models.generic_model import GenericModel, DESCRIPTION
from doors_detection_long_term.doors_detector.models.model_names import ModelName
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

TEST: DESCRIPTION = 0
class BboxFilterNetwork(GenericModel):
    def __init__(self, model_name: ModelName, pretrained: bool, num_bboxes: int, dataset_name: DATASET, description: DESCRIPTION):
        super(BboxFilterNetwork, self).__init__(model_name, dataset_name, description)
        self._num_bboxes = num_bboxes

        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.backbone.layer4 = nn.Sequential(*list(self.backbone.layer4.children())[:-2])
        self.backbone.fc = nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=self._num_bboxes,
                kernel_size=(3, 5)
            ),
            nn.BatchNorm2d(num_features=self._num_bboxes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 5)),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self._num_bboxes,
                out_channels=self._num_bboxes,
                kernel_size=(3, 5)
            ),
            nn.BatchNorm2d(num_features=self._num_bboxes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 5)),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=self._num_bboxes,
                out_channels=self._num_bboxes,
                kernel_size=(3, 5)
            ),
            nn.BatchNorm2d(num_features=self._num_bboxes),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 5)),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=self._num_bboxes,
                out_channels=self._num_bboxes,
                kernel_size=3
            ),
            nn.BatchNorm1d(num_features=self._num_bboxes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(
                in_channels=self._num_bboxes,
                out_channels=self._num_bboxes,
                kernel_size=3
            ),
            nn.BatchNorm1d(num_features=self._num_bboxes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Sigmoid()
        )

        if pretrained:
            if pretrained:
                path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, images, boxes):
        x = self.backbone(images)
        x = x.repeat(1, 64)
        x = torch.reshape(x, (*(x.size()[:-1]), 1, 64, 2048))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(2)
        x = torch.cat((x, boxes), 2)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.output_layer(x).squeeze(2)
        return x

class BboxFilterNetworkLoss(nn.Module):

    def __init__(self, weight=1.0, reduction_image='sum', reduction_global='mean'):
        super(BboxFilterNetworkLoss, self).__init__()
        self._weight = weight
        if not (reduction_image == 'sum' or reduction_image == 'mean'):
            raise Exception('Parameter "reduction_image" must be mean|sum')
        if not (reduction_global == 'sum' or reduction_global == 'mean'):
            raise Exception('Parameter "reduction_global" must be mean|sum')
        self._reduction_image = reduction_image
        self._reduction_global = reduction_global

    def forward(self, preds, targets):
        temp = -self._weight * (targets * torch.log(preds) + (1.0 - targets) * torch.log(1.0 - preds))


        if self._reduction_image == 'mean':
            reduction_on_image = torch.mean(temp, dim=1)
        elif self._reduction_image == 'sum':
            reduction_on_image = torch.sum(temp, dim=1)


        if self._reduction_global == 'mean':
            return torch.mean(reduction_on_image)
        elif self._reduction_global == 'sum':
            return torch.sum(reduction_on_image)

