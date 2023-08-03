from collections import OrderedDict

import cv2
import torch.optim
from matplotlib import pyplot as plt
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.ops import FeaturePyramidNetwork
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetsCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.bbox_filter_network import SharedMLP, TEST_IMAGE_LOCAL_NETWORK, \
    TEST_IMAGE_LOCAL_NETWORK_SMALL
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.doors_detector.utilities.util.bboxes_fintering import bounding_box_filtering_yolo, \
    check_bbox_dataset, plot_results
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
torch.autograd.detect_anomaly(True)
colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 40

iou_threshold_matching = 0.5
confidence_threshold = 0.75

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
        #ordered_dict['x4'] = x4

        #pyramid_features = self.fpn(ordered_dict)

        return x2

class BboxFilterNetworkImage(GenericModel):
    def __init__(self, fpn_channels:int, model_name: ModelName, pretrained: bool, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(BboxFilterNetworkImage, self).__init__(model_name, dataset_name, description)

        self.fpn = ResNet50FPN(channels=fpn_channels)
        self.batch_norm_after_fpn = nn.BatchNorm2d(fpn_channels)
        self.conv_after_backbone = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.shared_mlp_1 = SharedMLP(channels=[32, 32, 64, 128])
        self.shared_mlp_2 = SharedMLP(channels=[128, 256, 512, 1024])
        self.shared_mlp_3 = SharedMLP(channels=[512, 512, 1024])

        self.shared_mlp_4 = SharedMLP(channels=[1024 + 128, 256, 128, 64])

        self.shared_mlp_5 = SharedMLP(channels=[64, 32, 1], last_activation=nn.Sigmoid())

        self.shared_mlp_6 = SharedMLP(channels=[64, 32, n_labels], last_activation=nn.Softmax(dim=1))



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
        #print(x.size())
        x = self.conv_after_backbone(x)

        # Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2]
        converted_boxes = torch.cat([boxes[:, 0:1, :] - boxes[:, 2:3, :] / 2,
                                     boxes[:, 1:2, :] - boxes[:, 3:4, :] / 2,
                                     boxes[:, 0:1, :] + boxes[:, 2:3, :] / 2,
                                     boxes[:, 1:2, :] + boxes[:, 3:4, :] / 2], dim=1).transpose(1, 2)

        converted_boxes = torch.round(converted_boxes * torch.tensor([x.size()[2:][::-1] + x.size()[2:][::-1]], device=x.device))
        converted_boxes[:,:,0:1][[converted_boxes[:, :, 0:1] == converted_boxes[:, :, 2:3]]] -= 1
        converted_boxes[:,:,2:3][[converted_boxes[:, :, 0:1] == converted_boxes[:, :, 2:3]]] += 1
        converted_boxes[:,:,1:2][[converted_boxes[:, :, 1:2] == converted_boxes[:, :, 3:4]]] -= 1
        converted_boxes[:,:,3:4][[converted_boxes[:, :, 1:2] == converted_boxes[:, :, 3:4]]] += 1
        converted_boxes[converted_boxes <= 0.0] = 0.0
        converted_boxes[:,:,0:1][[converted_boxes[:, :, 0:1] >= x.size(-1)]] = x.size(-1)
        converted_boxes[:,:,2:3][[converted_boxes[:, :, 2:3] >= x.size(-1)]] = x.size(-1)
        converted_boxes[:,:,1:2][[converted_boxes[:, :, 1:2] >= x.size(-2)]] = x.size(-2)
        converted_boxes[:,:,3:4][[converted_boxes[:, :, 3:4] >= x.size(-2)]] = x.size(-2)

        converted_boxes = converted_boxes.type(torch.int16)

        local_features_bboxes_image = []


        for n_batch, bboxes_on_batch in enumerate(converted_boxes):
            local_features_bboxes_batch = []
            for x1, y1, x2, y2 in bboxes_on_batch:
                #print(x[n_batch, :, y1: y2, x1: x2].size())
                pooled = torch.mean(x[n_batch, :, y1: y2, x1: x2].clone(), dim=(1, 2))
                #print(pooled.size())
                #pooled[pooled == float('nan')] = 0.0
                #4print(pooled)
                local_features_bboxes_batch.append(pooled)
            local_features_bboxes_image.append(torch.stack(local_features_bboxes_batch))

        local_features_bboxes_image = torch.stack(local_features_bboxes_image).transpose(1, 2)

        local_features_1 = self.shared_mlp_1(local_features_bboxes_image)
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

        score_features = torch.squeeze(score_features, dim=1)
        label_features = torch.transpose(label_features, 1, 2)
        #print(score_features, label_features)
        return score_features, label_features


class BboxFilterNetworkGeometricLabelLoss(nn.Module):

    def __init__(self, weight=1.0, reduction_image='sum', reduction_global='mean'):
        super(BboxFilterNetworkGeometricLabelLoss, self).__init__()
        self._weight = weight
        if not (reduction_image == 'sum' or reduction_image == 'mean'):
            raise Exception('Parameter "reduction_image" must be mean|sum')
        if not (reduction_global == 'sum' or reduction_global == 'mean'):
            raise Exception('Parameter "reduction_global" must be mean|sum')
        self._reduction_image = reduction_image
        self._reduction_global = reduction_global

    def forward(self, preds, label_targets):
        scores_features, labels_features = preds
        labels_loss = torch.log(labels_features) * label_targets * torch.tensor([[0.5, 0.25, 0.25]], device=label_targets.device)
        labels_loss = torch.mean(torch.sum(torch.sum(labels_loss, 2) * -1, 1))

        return labels_loss

class BboxFilterNetworkGeometricConfidenceLoss(nn.Module):
    def forward(self, preds, confidences):
        scores_features, labels_features = preds
        confidence_loss = torch.mean(torch.sum(torch.abs(scores_features - confidences), dim=1))

        return confidence_loss


#images = torch.rand(32, 3, 240, 320)
#bboxes = torch.rand(32, 7, 20)
#bbox_model = BboxFilterNetworkImage(fpn_channels=32, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST_IMAGE_LOCAL_NETWORK_SMALL)
#bbox_model(images, bboxes)


dataset_creator_bboxes = DatasetsCreatorBBoxes()
dataset_creator_bboxes.load_dataset(folder_name='yolov5_simulation_dataset')
dataset_creator_bboxes.select_n_bounding_boxes(num_bboxes=num_bboxes)
dataset_creator_bboxes.match_bboxes_with_gt(iou_threshold_matching=iou_threshold_matching)

train_bboxes, test_bboxes = dataset_creator_bboxes.create_datasets(shuffle_boxes=False)

train_dataset_bboxes = DataLoader(train_bboxes, batch_size=16, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4, shuffle=False)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=16, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4)
#check_bbox_dataset(train_dataset_bboxes, confidence_threshold=confidence_threshold)

# Calculate Metrics in real worlds
houses = ['floor1', 'floor4', 'chemistry_floor0']

data_loaders_real_word = {}
labels = None
for house in houses:
    _, test, l, _ = get_final_doors_dataset_real_data(folder_name=house, train_size=0.25)
    labels = l
    data_loader_test = DataLoader(test, batch_size=4, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
    data_loaders_real_word[house] = data_loader_test

yolo_gd = YOLOv5Model(model_name=YOLOv5, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=globals()[f'EXP_GENERAL_DETECTOR_GIBSON_60_EPOCHS'.upper()])
yolo_gd.to('cuda')
yolo_gd.eval()

performances_in_real_worlds = {'AP': {'0': [], '1': []},
                               'TP': [], 'FP': [], 'TPm': [], 'FPiou': []}

datasets_real_worlds = {}
with torch.no_grad():
    for house in houses:
        dataset_creator_bboxes_real_world = DatasetsCreatorBBoxes()
        evaluator = MyEvaluator()
        evaluator_complete_metric = MyEvaluatorCompleteMetric()
        for images, targets, converted_boxes in tqdm(data_loaders_real_word[house], total=len(data_loaders_real_word[house]), desc=f'Evaluating yolo GD in {house}'):
            images = images.to('cuda')
            preds, train_out = yolo_gd.model(images)
            dataset_creator_bboxes_real_world.add_yolo_bboxes(images=images, targets=targets, preds=preds, bboxes_type=ExampleType.TEST)
            preds = bounding_box_filtering_yolo(preds, max_detections=300, iou_threshold=iou_threshold_matching, confidence_threshold=0.01, apply_nms=True)
            #preds = non_max_suppression(preds,0.01,0.5,multi_label=False, agnostic=True,max_det=300)
            evaluator.add_predictions_yolo(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])
            evaluator_complete_metric.add_predictions_yolo(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])
        metric = evaluator.get_metrics(iou_threshold=iou_threshold_matching, confidence_threshold=confidence_threshold)
        metric_complete = evaluator_complete_metric.get_metrics(iou_threshold=iou_threshold_matching, confidence_threshold=confidence_threshold)
        for label, values in sorted(metric['per_bbox'].items(), key=lambda v: v[0]):
            performances_in_real_worlds['AP'][label].append(values['AP'])

        temp = {'TP': [], 'FP': [], 'TPm': [], 'FPiou': []}
        for label, values in sorted(metric_complete.items(), key=lambda v: v[0]):
            temp['TP'].append(values['TP'])
            temp['FP'].append(values['FP'])
            temp['TPm'].append(values['TPm'])
            temp['FPiou'].append(values['FPiou'])
        performances_in_real_worlds['TP'].append(sum(temp['TP']))
        performances_in_real_worlds['FP'].append(sum(temp['FP']))
        performances_in_real_worlds['TPm'].append(sum(temp['TPm']))
        performances_in_real_worlds['FPiou'].append(sum(temp['FPiou']))

        dataset_creator_bboxes_real_world.select_n_bounding_boxes(num_bboxes=num_bboxes)
        dataset_creator_bboxes_real_world.match_bboxes_with_gt(iou_threshold_matching=iou_threshold_matching)

        #dataset_creator_bboxes_real_world.visualize_bboxes(bboxes_type=ExampleType.TEST)
        _, test_bboxes = dataset_creator_bboxes_real_world.create_datasets(shuffle_boxes=False, apply_transforms_to_train=False)
        datasets_real_worlds[house] = DataLoader(test_bboxes, batch_size=1, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4, shuffle=False)


performances_in_real_worlds['AP']['0'] = sum(performances_in_real_worlds['AP']['0']) / len(performances_in_real_worlds['AP']['0'])
performances_in_real_worlds['AP']['1'] = sum(performances_in_real_worlds['AP']['1']) / len(performances_in_real_worlds['AP']['1'])
performances_in_real_worlds['TP'] = sum(performances_in_real_worlds['TP'])
performances_in_real_worlds['FP'] = sum(performances_in_real_worlds['FP'])
performances_in_real_worlds['TPm'] = sum(performances_in_real_worlds['TPm'])
performances_in_real_worlds['FPiou'] = sum(performances_in_real_worlds['FPiou'])


print(performances_in_real_worlds)

# Plot evaluation in real worlds
fig = plt.figure()
plt.axhline(y=performances_in_real_worlds['AP']['0'], color = 'r', linestyle = '--', label='closed doors')
plt.axhline(y=performances_in_real_worlds['AP']['1'], color = 'g', linestyle = '--', label='open doors')
plt.title('AP')
plt.legend()
plt.savefig('image_global_net/AP.svg')

fig = plt.figure()
plt.axhline(y=performances_in_real_worlds['TP'], color = 'g', linestyle = '--', label='TP')
plt.axhline(y=performances_in_real_worlds['FP'], color = 'r', linestyle = '--', label='FP')
plt.axhline(y=performances_in_real_worlds['TPm'], color = 'forestgreen', linestyle = '--', label='TPm')
plt.axhline(y=performances_in_real_worlds['FPiou'], color = 'salmon', linestyle = '--', label='FPiou')
plt.title('Complete metric')
plt.legend()
plt.savefig('image_global_net/complete_metric.svg')

#check_bbox_dataset(datasets_real_worlds['floor4'], confidence_threshold)
bbox_model = BboxFilterNetworkImage(fpn_channels=32, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST_IMAGE_LOCAL_NETWORK_SMALL)
bbox_model.to('cuda')

criterion_label = BboxFilterNetworkGeometricLabelLoss(reduction_image='sum', reduction_global='mean')
criterion_confidence = BboxFilterNetworkGeometricConfidenceLoss()

optimizer = optim.Adam(bbox_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion_label.to('cuda')
criterion_confidence.to('cuda')
#for n, p in bbox_model.named_parameters():
#    if p.requires_grad:
#        print(n)

logs = {'train': {'loss_label':[], 'loss_confidence':[], 'loss_final':[]}, 'test': {'loss_label':[], 'loss_confidence':[], 'loss_final':[]}, 'ap': {0: [], 1: []}, 'complete_metric': {'TP': [], 'FP': [], 'BFD': []}}

# compute total labels
train_total = {0:0, 1:0, 2:0}
test_total = {0:0, 1:0, 2:0}
for data in train_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
    for labels in labels_encoded:
        for label in labels:
            train_total[int((label == 1).nonzero(as_tuple=True)[0])] += 1

for data in test_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
    for labels in labels_encoded:
        for label in labels:
            test_total[int((label == 1).nonzero(as_tuple=True)[0])] += 1

train_accuracy = {0: [], 1: [], 2: []}
test_accuracy = {0: [], 1: [], 2: []}
performances_in_real_worlds_bbox_filtering = {'AP': {'0': [], '1': []},
                                              'TP': [], 'FP': [], 'TPm': [], 'FPiou': []}
for epoch in range(60):
    #scheduler.step()
    bbox_model.train()
    criterion_label.train()
    optimizer.zero_grad()

    temp_losses_label = []
    temp_losses_confidence = []
    temp_losses_final = []

    for i, data in tqdm(enumerate(train_dataset_bboxes), total=len(train_dataset_bboxes), desc=f'Training epoch {epoch}'):
        if i > 2902:
            print(detected_bboxes)

        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
        images = images.to('cuda')
        detected_bboxes = detected_bboxes.to('cuda')
        confidences = confidences.to('cuda')
        labels_encoded = labels_encoded.to('cuda')
        ious = ious.to('cuda')

        preds = bbox_model(images, detected_bboxes)
        #print(preds, filtered)
        loss_label = criterion_label(preds, labels_encoded)
        loss_confidence = criterion_confidence(preds, ious)
        final_loss = loss_label + loss_confidence


        temp_losses_label.append(loss_label.item())
        temp_losses_confidence.append(loss_confidence.item())
        temp_losses_final.append(final_loss.item())
        optimizer.zero_grad()
        final_loss.backward()
        #official_loss.backward()
        optimizer.step()
    logs['train']['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))
    logs['train']['loss_confidence'].append(sum(temp_losses_confidence) / len(temp_losses_confidence))
    logs['train']['loss_label'].append(sum(temp_losses_label) / len(temp_losses_label))

    temp_losses_label = []
    temp_losses_confidence = []
    temp_losses_final = []

    with torch.no_grad():

        bbox_model.eval()
        criterion_label.eval()
        criterion_confidence.eval()

        temp_accuracy = {0:0, 1:0, 2:0}
        for data in tqdm(train_dataset_bboxes, total=len(train_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
            images = images.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')
            ious = ious.to('cuda')

            preds = bbox_model(images, detected_bboxes)

            predicted_labels = preds[1]

            for predicted, target in zip(predicted_labels, labels_encoded):
                values, p_indexes = torch.max(predicted, dim=1)
                values, gt_indexes = torch.max(target, dim=1)
                for p_index, gt_index in zip(p_indexes.tolist(), gt_indexes.tolist()):
                    if gt_index == p_index:
                        temp_accuracy[int(gt_index)] += 1

        for i in range(3):
            train_accuracy[i].append(temp_accuracy[i] / train_total[i])

        temp_accuracy = {0:0, 1:0, 2:0}

        for data in tqdm(test_dataset_bboxes, total=len(test_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
            images = images.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')
            ious = ious.to('cuda')

            preds = bbox_model(images, detected_bboxes)
            loss_label = criterion_label(preds, labels_encoded)
            loss_confidence = criterion_confidence(preds, ious)
            final_loss = loss_label + loss_confidence

            temp_losses_final.append(final_loss.item())
            temp_losses_confidence.append(loss_confidence.item())
            temp_losses_label.append(loss_label.item())

            predicted_labels = preds[1]

            for predicted, target in zip(predicted_labels, labels_encoded):
                values, p_indexes = torch.max(predicted, dim=1)
                values, gt_indexes = torch.max(target, dim=1)
                for p_index, gt_index in zip(p_indexes.tolist(), gt_indexes.tolist()):
                    if gt_index == p_index:
                        temp_accuracy[int(gt_index)] += 1

        for i in range(3):
            test_accuracy[i].append(temp_accuracy[i] / test_total[i])

    # Test with real world data

    temp = {'AP':{'0':[], '1':[]}, 'TP': [], 'FP': [], 'TPm': [], 'FPiou': []}
    for house, dataset_real_world in datasets_real_worlds.items():
        evaluator = MyEvaluator()
        evaluator_complete_metric = MyEvaluatorCompleteMetric()
        for c, data in tqdm(enumerate(dataset_real_world), total=len(dataset_real_world), desc=f'TEST in {house}, epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
            images = images.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')
            ious = ious.to('cuda')

            preds = bbox_model(images, detected_bboxes)

            plot_results(epoch=epoch, count=c, env=house, images=images, bboxes=detected_bboxes, preds=preds, targets=target_boxes, confidence_threshold = confidence_threshold)

            evaluator.add_predictions_bboxes_filtering(detected_bboxes, preds, target_boxes, img_size=images.size()[2:][::-1])
            evaluator_complete_metric.add_predictions_bboxes_filtering(detected_bboxes, preds, target_boxes, img_size=images.size()[2:][::-1])

        metric = evaluator.get_metrics(iou_threshold=iou_threshold_matching, confidence_threshold=confidence_threshold)
        metric_complete = evaluator_complete_metric.get_metrics(iou_threshold=iou_threshold_matching, confidence_threshold=confidence_threshold)

        for label, values in sorted(metric['per_bbox'].items(), key=lambda v: v[0]):
            temp['AP'][label].append(values['AP'])

        for label, values in sorted(metric_complete.items(), key=lambda v: v[0]):
            temp['TP'].append(values['TP'])
            temp['FP'].append(values['FP'])
            temp['TPm'].append(values['TPm'])
            temp['FPiou'].append(values['FPiou'])

    #performances_in_real_worlds_bbox_filtering['AP']['0'].append(sum(temp['AP']['0']) / len(temp['AP']['0']))
    performances_in_real_worlds_bbox_filtering['AP']['1'].append(sum(temp['AP']['1']) / len(temp['AP']['1']))
    performances_in_real_worlds_bbox_filtering['TP'].append(sum(temp['TP']))
    performances_in_real_worlds_bbox_filtering['FP'].append(sum(temp['FP']))
    performances_in_real_worlds_bbox_filtering['TPm'].append(sum(temp['TPm']))
    performances_in_real_worlds_bbox_filtering['FPiou'].append(sum(temp['FPiou']))

    # Plot evaluation in real worlds
    fig = plt.figure()
    plt.axhline(y=performances_in_real_worlds['AP']['0'], color = 'r', linestyle = '--', label='closed doors')
    plt.axhline(y=performances_in_real_worlds['AP']['1'], color = 'g', linestyle = '--', label='open doors')
    plt.title('AP')
    plt.legend()
    plt.savefig('AP.svg')

    fig = plt.figure()
    plt.axhline(y=performances_in_real_worlds['TP'], color = 'g', linestyle = '--', label='TP')
    plt.axhline(y=performances_in_real_worlds['FP'], color = 'r', linestyle = '--', label='FP')
    plt.axhline(y=performances_in_real_worlds['TPm'], color = 'forestgreen', linestyle = '--', label='TPm')
    plt.axhline(y=performances_in_real_worlds['FPiou'], color = 'salmon', linestyle = '--', label='FPiou')
    plt.plot([i for i in range(len(performances_in_real_worlds_bbox_filtering['TP']))], performances_in_real_worlds_bbox_filtering['TP'], label='TP')
    plt.plot([i for i in range(len(performances_in_real_worlds_bbox_filtering['FP']))], performances_in_real_worlds_bbox_filtering['FP'], label='FP')
    plt.plot([i for i in range(len(performances_in_real_worlds_bbox_filtering['TPm']))], performances_in_real_worlds_bbox_filtering['TPm'], label='TPm')
    plt.plot([i for i in range(len(performances_in_real_worlds_bbox_filtering['FPiou']))], performances_in_real_worlds_bbox_filtering['FPiou'], label='FPiou')

    plt.title('Complete metric')
    plt.legend()
    plt.savefig('image_global_net/complete_metric.svg')

    print(performances_in_real_worlds_bbox_filtering)
    print(train_accuracy)
    fig = plt.figure()
    plt.plot([i for i in range(len(train_accuracy[0]))], train_accuracy[0], label='background')
    plt.plot([i for i in range(len(train_accuracy[1]))], train_accuracy[1], label='closed')
    plt.plot([i for i in range(len(train_accuracy[2]))], train_accuracy[2], label='open')
    plt.title('Train accuracy')
    plt.legend()
    plt.savefig('image_global_net/train_geometric.svg')

    fig = plt.figure()
    plt.plot([i for i in range(len(test_accuracy[0]))], test_accuracy[0], label='background')
    plt.plot([i for i in range(len(test_accuracy[1]))], test_accuracy[1], label='closed')
    plt.plot([i for i in range(len(test_accuracy[2]))], test_accuracy[2], label='open')
    plt.title('Val accuracy')
    plt.legend()
    plt.savefig('image_global_net/val_geometric.svg')
    print(test_accuracy)
    logs['test']['loss_final'].append(sum(temp_losses_final) / len(temp_losses_final))
    logs['test']['loss_confidence'].append(sum(temp_losses_confidence) / len(temp_losses_confidence))
    logs['test']['loss_label'].append(sum(temp_losses_label) / len(temp_losses_label))
    print(logs['train'], logs['test'])

    fig = plt.figure()
    plt.plot([i for i in range(len(logs['train']['loss_label']))], logs['train']['loss_label'], label='train_loss')
    plt.plot([i for i in range(len(logs['train']['loss_label']))], logs['test']['loss_label'], label='test_loss')
    plt.title('Losses')
    plt.legend()
    plt.savefig('image_global_net/losses_geometric_label.svg')

    fig = plt.figure()
    plt.plot([i for i in range(len(logs['train']['loss_confidence']))], logs['train']['loss_confidence'], label='train_loss')
    plt.plot([i for i in range(len(logs['train']['loss_confidence']))], logs['test']['loss_confidence'], label='test_loss')
    plt.title('Losses')
    plt.legend()
    plt.savefig('image_global_net/losses_geometric_confidence.svg')
    bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})










