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

        if last_activation is None:
            layers = layers[:-2]
        else:
            layers[-1] = last_activation
        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.shared_mlp(input)


class MaskNetwork(nn.Module):
    def __init__(self, image_size: Tuple[int, int]):
        super(MaskNetwork, self).__init__()

        self._image_size = image_size

        final_size = image_size[0] * image_size[1]
        self.x1 = nn.Linear(in_features=1, out_features=final_size)
        self.x2 = nn.Linear(in_features=1, out_features=final_size)
        self.y1 = nn.Linear(in_features=1, out_features=final_size)
        self.y2 = nn.Linear(in_features=1, out_features=final_size)

        with torch.no_grad():
            self.x1.weight.fill_(1.0)
            self.x2.weight.fill_(1.0)
            self.y1.weight.fill_(1.0)
            self.y2.weight.fill_(1.0)
            self.x1.bias = nn.Parameter(torch.tensor([i % image_size[0] * -1.0 for i in range(final_size)]))
            self.x2.bias = nn.Parameter(torch.tensor([i % image_size[0] * -1.0 for i in range(final_size)]))
            self.y1.bias = nn.Parameter(torch.tensor([i % image_size[0] * -1.0 for i in range(final_size)]).reshape(image_size).transpose(0,1).flatten())
            self.y2.bias = nn.Parameter(torch.tensor([i % image_size[0] * -1.0 for i in range(final_size)]).reshape(image_size).transpose(0,1).flatten())

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, bboxes_masks, features_number: int = 1):

        x1 = self.x1(bboxes_masks[:, 0].view(-1, 1))
        x2 = self.x2(bboxes_masks[:, 2].view(-1, 1))
        y1 = self.y1(bboxes_masks[:, 1].view(-1, 1))
        y2 = self.y2(bboxes_masks[:, 3].view(-1, 1))

        x1 = x1 <= 0
        x2 = x2 > 0
        y1 = y1 <= 0
        y2 = y2 > 0
        x = x1 * x2 * y1 * y2
        x = x.reshape(bboxes_masks.size(0), 1, self._image_size[0], self._image_size[1]).repeat(1, features_number, 1, 1)

        return x


class BboxFilterNetworkGeometricBackground(GenericModel):
    def __init__(self, model_name: ModelName, pretrained: bool, initial_channels: int, n_labels: int, dataset_name: DATASET, description: DESCRIPTION, description_background: DESCRIPTION, image_grid_dimensions: List[Tuple[int, int]]):
        super(BboxFilterNetworkGeometricBackground, self).__init__(model_name, dataset_name, description)
        self._image_grid_dimensions = image_grid_dimensions
        self._initial_channels = initial_channels

        self.background_network = ImageGridNetwork(fpn_channels=256, image_grid_dimensions=image_grid_dimensions, n_labels=3, model_name=IMAGE_BACKGROUND_NETWORK, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=description_background)

        self.background_network.final_convolution = nn.Sequential(*self.background_network.final_convolution[:-3])

        self.mask_network = MaskNetwork(image_size=image_grid_dimensions[0])

        self.shared_mlp_background_1 = SharedMLP(channels=[16, 32, 64, 128])
        self.shared_mlp_background_2 = SharedMLP(channels=[128, 256, 512, 1024])

        self.shared_mlp_mix_background = SharedMLP(channels=[1024+128, 1024, 512, 256])
        self.shared_mlp_suppress_background = SharedMLP(channels=[256, 128, 64, 32, 16, 1], last_activation=nn.Sigmoid())

        # Geometric
        self.shared_mlp_geometric_1 = SharedMLP(channels=[initial_channels, 16, 32, 64, 128])
        self.shared_mlp_geometric_2 = SharedMLP(channels=[128, 256, 512, 1024])
        self.shared_mlp_mix_geometric = SharedMLP(channels=[1024+128, 1024, 512, 256])
        self.shared_mlp_new_labels = SharedMLP(channels=[256, 128, 64, 32, 16, n_labels], last_activation=nn.Softmax(dim=1))

        # Mixed
        self.shared_mlp_new_confidences = SharedMLP(channels=[512, 256, 128, 64, 32, 10], last_activation=nn.Softmax(dim=1))

        if pretrained:
            if pretrained:
                path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, images, bboxes, bboxes_mask):

        # Background
        background_features = self.background_network(images).transpose(1, 2).transpose(2, 3)
        mask = self.mask_network(bboxes_mask[self._image_grid_dimensions[0]].view(-1, 4).float(), background_features.size(1))
        mask = mask.view(images.size(0), bboxes.size(-1), *mask.size()[1:])
        # Max
        background_features = background_features.unsqueeze(1)
        bounding_boxes_background_features = torch.amax(mask * background_features, dim=(3, 4)).transpose(1, 2)
        # Mean
        #background_features = background_features.unsqueeze(1)
        #bounding_boxes_background_features = ((mask * background_features).sum(dim=(3, 4)) / mask.sum(dim=(3, 4))).transpose(1, 2)

        local_features_background = self.shared_mlp_background_1(bounding_boxes_background_features)
        global_features_background = self.shared_mlp_background_2(local_features_background)
        global_features_background = torch.max(global_features_background, 2, keepdim=True)[0]
        global_features_background = global_features_background.repeat(1, 1, local_features_background.size(-1))

        mixed_features_background = torch.cat([local_features_background, global_features_background], dim=1)
        mixed_features_background = self.shared_mlp_mix_background(mixed_features_background)

        # Output suppress background
        suppress_background = self.shared_mlp_suppress_background(mixed_features_background)
        suppress_background = torch.squeeze(suppress_background, dim=1)

        # Geometric part
        local_features_geometric = self.shared_mlp_geometric_1(bboxes)
        global_features_geometric = self.shared_mlp_geometric_2(local_features_geometric)

        global_features_geometric = torch.max(global_features_geometric, 2, keepdim=True)[0]
        global_features_geometric = global_features_geometric.repeat(1, 1, local_features_geometric.size(-1))
        mixed_features_geometric = torch.cat([local_features_geometric, global_features_geometric], 1)
        mixed_features_geometric = self.shared_mlp_mix_geometric(mixed_features_geometric)

        # New labels output
        new_labels = self.shared_mlp_new_labels(mixed_features_geometric)
        new_labels = torch.transpose(new_labels, 1, 2)

        # Mixed part
        mixed_features_geometric_background = torch.cat([mixed_features_geometric, mixed_features_background], dim=1)

        # Output new confidences
        new_confidences = self.shared_mlp_new_confidences(mixed_features_geometric_background)
        new_confidences = torch.transpose(new_confidences, 1, 2)

        return new_labels, suppress_background, new_confidences


class BboxFilterNetworkGeometricLabelLoss(nn.Module):

    def forward(self, labels_features, label_targets):

        labels_loss = torch.log(labels_features) * label_targets #* torch.tensor([[0.20, 1, 1]], device='cuda')
        labels_loss = torch.mean(torch.mean(torch.sum(labels_loss, 2) * -1, 1))

        return labels_loss


class BboxFilterNetworkGeometricSuppressLoss(nn.Module):
    def forward(self, suppress_features, confidences):

        #confidence_loss = torch.mean(torch.mean(torch.abs(scores_features - confidences), dim=1))
        #print(-torch.sum(torch.log(scores_features) * confidences + torch.log(1-scores_features) * (1-confidences), dim=1).size())
        confidence_loss = torch.mean(-torch.mean(torch.log(suppress_features) * confidences + torch.log(1-suppress_features) * (1-confidences), dim=1))
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

"""
mask_network = MaskNetwork(image_size=(4, 4))
print(mask_network(torch.rand(30*16, 4)))

"""