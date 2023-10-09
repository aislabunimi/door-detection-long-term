import os
from typing import List, Tuple

import torch
import torchvision.ops
from torch import nn

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.background_grid_network import ImageGridNetwork
from doors_detection_long_term.doors_detector.models.generic_model import DESCRIPTION, GenericModel
from doors_detection_long_term.doors_detector.models.model_names import ModelName, IMAGE_BACKGROUND_NETWORK
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

TEST: DESCRIPTION = 0
IMAGE_NETWORK_GEOMETRIC_BACKGROUND: DESCRIPTION = 1

class SharedMLP(nn.Module):
    def __init__(self, channels, last_activation: nn.Module = nn.ReLU()):
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


class BboxFilterNetworkGeometricBackground(GenericModel):
    def __init__(self, model_name: ModelName, pretrained: bool, initial_channels: int, n_labels: int, dataset_name: DATASET, description: DESCRIPTION, description_background: DESCRIPTION, image_grid_dimensions: List[Tuple[int, int]]):
        super(BboxFilterNetworkGeometricBackground, self).__init__(model_name, dataset_name, description)
        self._initial_channels = initial_channels

        self.background_network = ImageGridNetwork(fpn_channels=256, image_grid_dimensions=image_grid_dimensions, n_labels=3, model_name=IMAGE_BACKGROUND_NETWORK, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=description_background)

        self.background_network.final_convolution = nn.Sequential(*self.background_network.final_convolution[:-3], nn.ReLU())

        self.background_to_geometric = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.Sigmoid(),
        )

        self.shared_mlp_1 = SharedMLP(channels=[initial_channels, 32, 64, 128, 256])
        self.shared_mlp_2 = SharedMLP(channels=[256, 256, 512, 1024])
        self.shared_mlp_3 = SharedMLP(channels=[1024, 1024, 2048])

        self.shared_mlp_4 = SharedMLP(channels=[256 + 1024 + 2048, 2048, 1024, 512, 256, 128])

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

    def forward(self, images, bboxes, bboxes_mask):

        #background_features = torch.transpose(self.background_to_geometric(self.background_network(images)), 2, 3)
        #print(background_features.size(), background_features.size()[::-1][:2], bboxes_mask[background_features.size()[::-1][:2]])
        #bboxes_mask = bboxes_mask[background_features.size()[::-1][:2]]

        bboxes_background_features = []
        """
        for features, bboxes_mask_list in zip(background_features, bboxes_mask):
            for x1, y1, x2, y2 in bboxes_mask_list:
                mean = torch.mean(features[:, x1: x2, y1: y2].reshape(background_features.size()[1], -1), dim=1)
                bboxes_background_features.append(mean)
        bboxes = torch.cat([bboxes, torch.stack(bboxes_background_features).reshape(background_features.size()[0], background_features.size()[1], -1)], dim=1)
        """
        local_features_1 = self.shared_mlp_1(bboxes)
        local_features_2 = self.shared_mlp_2(local_features_1)
        local_features_3 = self.shared_mlp_3(local_features_2)

        global_features_1 = torch.max(local_features_2, 2, keepdim=True)[0]
        global_features_1 = global_features_1.repeat(1, 1, local_features_1.size(-1))

        global_features_2 = torch.max(local_features_3, 2, keepdim=True)[0]
        global_features_2 = global_features_2.repeat(1, 1, local_features_1.size(-1))

        mixed_features = torch.cat([local_features_1, global_features_1, global_features_2], 1)

        mixed_features = self.shared_mlp_4(mixed_features)

        score_features = self.shared_mlp_5(mixed_features)
        label_features = self.shared_mlp_6(mixed_features)

        score_features = torch.squeeze(score_features, dim=1)
        label_features = torch.transpose(label_features, 1, 2)

        return score_features, label_features


class BboxFilterNetworkGeometricLabelLoss(nn.Module):

    def forward(self, preds, label_targets):

        scores_features, labels_features = preds
        #print(labels_features, label_targets)
        labels_loss = torch.log(labels_features) * label_targets #* torch.tensor([[0.20, 1, 1]], device='cuda')
        #print(labels_loss)
        labels_loss = torch.mean(torch.mean(torch.sum(labels_loss, 2) * -1, 1))

        return labels_loss


class BboxFilterNetworkGeometricConfidenceLoss(nn.Module):
    def forward(self, preds, confidences):
        scores_features, labels_features = preds
        confidence_loss = torch.mean(torch.mean(-torch.log(1-torch.abs(scores_features - confidences)), dim=1))

        return confidence_loss


def bbox_filtering_nms(bboxes, img_size, iou_threshold=.1, confidence_threshold=0.75):

    filtered_bboxes = []
    for image_bboxes in bboxes:
        image_bboxes = image_bboxes[image_bboxes[:, 4] >= confidence_threshold]

        coords = torch.stack([image_bboxes[:, 0] - image_bboxes[:, 2] / 2,
                              image_bboxes[:, 1] - image_bboxes[:, 3] / 2,
                              image_bboxes[:, 0] + image_bboxes[:, 2] / 2,
                              image_bboxes[:, 1] + image_bboxes[:, 3] / 2], dim=1)
        coords[coords < 0.0] = 0.0
        coords = coords * torch.tensor([[img_size[0], img_size[1], img_size[0], img_size[1]]])
        keep = torchvision.ops.nms(boxes=coords, scores=image_bboxes[:, 4], iou_threshold=iou_threshold)
        filtered_bboxes.append(image_bboxes[keep])
    return filtered_bboxes