import os

import torch
import torchvision.models
from torch import nn
from torch.nn import Identity, AdaptiveAvgPool2d

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.generic_model import GenericModel, DESCRIPTION
from doors_detection_long_term.doors_detector.models.model_names import *
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

TEST: DESCRIPTION = 0

class SharedMLP(nn.Module):
    def __init__(self, channels, last_activation=nn.ReLU()):
        super(SharedMLP, self).__init__()
        layers = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            layers += [nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                       nn.BatchNorm1d(num_features=out_channels),
                       nn.ReLU()]
        layers[-1] = last_activation
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.shared_mlp(input)

class BboxFilterNetworkGeometric(GenericModel):
    def __init__(self, model_name: ModelName, pretrained: bool, initial_channels: int, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(BboxFilterNetworkGeometric, self).__init__(model_name, dataset_name, description)
        self._initial_channels = initial_channels

        self.shared_mlp_1 = SharedMLP(channels=[initial_channels, 16, 32, 64])
        self.shared_mlp_2 = SharedMLP(channels=[64, 64, 128, 256])
        self.shared_mlp_3 = SharedMLP(channels=[256, 256, 512, 1024])

        self.shared_mlp_4 = SharedMLP(channels=[1024 + 256 + 64, 512, 256, 128])

        self.shared_mlp_5 = SharedMLP(channels=[128, 64, 32, 16, 1], last_activation=nn.Sigmoid())

        self.shared_mlp_6 = SharedMLP(channels=[128, 64, 32, 16, n_labels], last_activation=nn.Softmax(dim=1))


        if pretrained:
            if pretrained:
                path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, images, bboxes):
        local_features_1 = self.shared_mlp_1(bboxes)
        local_features_2 = self.shared_mlp_2(local_features_1)
        local_features_3 = self.shared_mlp_3(local_features_2)

        global_features_1 = nn.MaxPool1d(local_features_2.size(-1))(local_features_2)
        global_features_1 = nn.Flatten(1)(global_features_1).unsqueeze(-1)
        global_features_1 = global_features_1.repeat(1, 1, local_features_1.size(-1))

        global_features_2 = nn.MaxPool1d(local_features_3.size(-1))(local_features_3)
        global_features_2 = nn.Flatten(1)(global_features_2).unsqueeze(-1)
        global_features_2 = global_features_2.repeat(1, 1, local_features_1.size(-1))

        mixed_features = torch.cat([local_features_1, global_features_1, global_features_2], 1)

        mixed_features = self.shared_mlp_4(mixed_features)

        score_features = self.shared_mlp_5(mixed_features)
        label_features = self.shared_mlp_6(mixed_features)

        score_features = torch.squeeze(score_features)
        label_features = torch.transpose(label_features, 1, 2)

        return score_features, label_features

class BboxFilterNetworkGeometricLoss(nn.Module):

    def __init__(self, weight=1.0, reduction_image='sum', reduction_global='mean'):
        super(BboxFilterNetworkGeometric, self).__init__()
        self._weight = weight
        if not (reduction_image == 'sum' or reduction_image == 'mean'):
            raise Exception('Parameter "reduction_image" must be mean|sum')
        if not (reduction_global == 'sum' or reduction_global == 'mean'):
            raise Exception('Parameter "reduction_global" must be mean|sum')
        self._reduction_image = reduction_image
        self._reduction_global = reduction_global

    def forward(self, preds, targets):
        scores_features, labels_features = preds
        label_targets = targets['label_targets']
        labels_loss = torch.log(labels_features) * label_targets
        labels_loss = torch.mean(-torch.sum(labels_loss, (2, 1)))

        return torch.tensor(0), labels_loss



bboxes = torch.rand((2, 7, 50))

model = BboxFilterNetworkGeometric(initial_channels=7, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST)

scores, labels = model(torch.tensor([]), bboxes)

print(scores.size(), labels.size(), labels)