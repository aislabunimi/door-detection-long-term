import os
from collections import OrderedDict

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
IMAGE_TEST = 1

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

class BboxFilterNetworkGeometric(GenericModel):
    def __init__(self, model_name: ModelName, pretrained: bool, initial_channels: int, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(BboxFilterNetworkGeometric, self).__init__(model_name, dataset_name, description)
        self._initial_channels = initial_channels

        self.shared_mlp_1 = SharedMLP(channels=[initial_channels, 16, 32])
        self.shared_mlp_2 = SharedMLP(channels=[32, 64, 128])
        self.shared_mlp_3 = SharedMLP(channels=[512, 512, 1024])

        self.shared_mlp_4 = SharedMLP(channels=[32 + 128, 128, 64])

        self.shared_mlp_5 = SharedMLP(channels=[64, 32, 16, 1], last_activation=nn.Sigmoid())

        self.shared_mlp_6 = SharedMLP(channels=[64, 32, 16, n_labels], last_activation=nn.Softmax(dim=1))


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
        #local_features_3 = self.shared_mlp_3(local_features_2)

        global_features_1 = torch.max(local_features_2, 2, keepdim=True)[0]
        global_features_1 = global_features_1.repeat(1, 1, local_features_1.size(-1))

        #global_features_2 = torch.max(local_features_3, 2, keepdim=True)[0]
        #global_features_2 = global_features_2.repeat(1, 1, local_features_1.size(-1))

        mixed_features = torch.cat([local_features_1, global_features_1], 1)

        mixed_features = self.shared_mlp_4(mixed_features)

        score_features = self.shared_mlp_5(mixed_features)
        label_features = self.shared_mlp_6(mixed_features)

        score_features = torch.squeeze(score_features)
        label_features = torch.transpose(label_features, 1, 2)

        return score_features, label_features

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


class BboxFilterNetworkImage(GenericModel):
    def __init__(self, fpn_channels:int, model_name: ModelName, pretrained: bool, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(BboxFilterNetworkImage, self).__init__(model_name, dataset_name, description)

        self.fpn = ResNet50FPN(channels=fpn_channels)

        self.shared_mlp_1 = SharedMLP(channels=[fpn_channels, 256])
        self.shared_mlp_2 = SharedMLP(channels=[256, 512, 1024])
        #self.shared_mlp_3 = SharedMLP(channels=[256, 256, 512, 1024])

        self.shared_mlp_4 = SharedMLP(channels=[1024 + 256, 1024, 512, 256, 128])

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

    def forward(self, images, boxes):
        x = self.fpn(images)
        x = x['x1']


        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        converted_boxes = torch.cat([boxes[:, 0:1, :] - boxes[:, 2:3, :] / 2,
                                     boxes[:, 1:2, :] - boxes[:, 3:4, :] / 2,
                                     boxes[:, 0:1, :] + boxes[:, 2:3, :] / 2,
                                     boxes[:, 1:2, :] + boxes[:, 3:4, :] / 2], dim=1).transpose(1, 2)

        converted_boxes = torch.round(converted_boxes * torch.tensor([x.size()[2:][::-1] + x.size()[2:][::-1]], device=x.device))
        converted_boxes[converted_boxes <= 0.0] = 0.0
        converted_boxes[converted_boxes[:, :, 0] >= x.size(-1)] = x.size(-1)
        converted_boxes[converted_boxes[:, :, 2] >= x.size(-1)] = x.size(-1)
        converted_boxes[converted_boxes[:, :, 1] >= x.size(-2)] = x.size(-2)
        converted_boxes[converted_boxes[:, :, 3] >= x.size(-2)] = x.size(-2)

        converted_boxes = converted_boxes.type(torch.int32)

        #print(converted_boxes)

        boxes_features = []

        for n_batch, batch in enumerate(converted_boxes):
            boxes_features_batch = []
            for n_box, (x1, y1, x2, y2) in enumerate(batch):
                mask = torch.zeros(x.size()[1:], device=x.device, requires_grad=False)
                mask[:, y1 : y2, x1 : x2] = 1.0
                box_features = x[n_batch] * mask
                box_features = torch.max(torch.max(box_features, dim=2)[0], dim=1)[0].unsqueeze(0)
                boxes_features_batch.append(box_features)
            boxes_features.append(torch.cat(boxes_features_batch, dim=0).unsqueeze(0))

        x = torch.cat(boxes_features, dim=0).transpose(1, 2)



        local_features_1 = self.shared_mlp_1(x)
        local_features_2 = self.shared_mlp_2(local_features_1)
        #local_features_3 = self.shared_mlp_3(local_features_2)

        global_features_1 = torch.max(local_features_2, 2, keepdim=True)[0]
        global_features_1 = global_features_1.repeat(1, 1, local_features_1.size(-1))

        #global_features_2 = nn.MaxPool1d(local_features_3.size(-1))(local_features_3)
        #global_features_2 = global_features_2.repeat(1, 1, local_features_1.size(-1))

        mixed_features = torch.cat([local_features_1, global_features_1], 1)

        mixed_features = self.shared_mlp_4(mixed_features)

        score_features = self.shared_mlp_5(mixed_features)
        label_features = self.shared_mlp_6(mixed_features)
        score_features = torch.squeeze(score_features, dim=1)
        label_features = torch.transpose(label_features, 1, 2)
        return score_features, label_features


class BboxFilterNetworkGeometricLoss(nn.Module):

    def __init__(self, weight=1.0, reduction_image='sum', reduction_global='mean'):
        super(BboxFilterNetworkGeometricLoss, self).__init__()
        self._weight = weight
        if not (reduction_image == 'sum' or reduction_image == 'mean'):
            raise Exception('Parameter "reduction_image" must be mean|sum')
        if not (reduction_global == 'sum' or reduction_global == 'mean'):
            raise Exception('Parameter "reduction_global" must be mean|sum')
        self._reduction_image = reduction_image
        self._reduction_global = reduction_global

    def forward(self, preds, confidences, label_targets):
        scores_features, labels_features = preds
        labels_loss = torch.log(labels_features) * label_targets #* torch.tensor([[0.15, 0.7, 0.15]], device=label_targets.device)
        labels_loss = torch.mean(torch.sum(torch.sum(labels_loss, 2) * -1, 1))

        return torch.tensor(0), labels_loss

class BboxFilterNetworkSuppress(nn.Module):

    def __init__(self):
        super(BboxFilterNetworkSuppress, self).__init__()

    def forward(self, preds, confidences, label_targets):
        scores_features, labels_features = preds
        scores_features = -(confidences * torch.log(scores_features) + ((1 - confidences) * torch.log(1-scores_features)))
        scores_features = torch.mean(scores_features, (1, 0))

        return scores_features, torch.tensor(0)


model = BboxFilterNetworkImage(fpn_channels=128, n_labels=3, model_name=BBOX_FILTER_NETWORK_IMAGE, description=IMAGE_TEST, dataset_name=FINAL_DOORS_DATASET, pretrained=False)
x = torch.rand(2, 3, 240, 320)
output = model(x, torch.rand(2, 7, 50))
print(output[0])


