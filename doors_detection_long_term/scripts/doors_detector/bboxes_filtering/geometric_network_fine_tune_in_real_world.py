import cv2
import torch.optim
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.bbox_filter_network import SharedMLP
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.doors_detector.utilities.util.bboxes_fintering import bounding_box_filtering_yolo, \
    check_bbox_dataset, plot_results
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 30

iou_threshold_matching = 0.5
confidence_threshold = 0.75

class BboxFilterNetworkGeometric(GenericModel):
    def __init__(self, model_name: ModelName, pretrained: bool, initial_channels: int, n_labels: int, dataset_name: DATASET, description: DESCRIPTION):
        super(BboxFilterNetworkGeometric, self).__init__(model_name, dataset_name, description)
        self._initial_channels = initial_channels

        self.shared_mlp_1 = SharedMLP(channels=[initial_channels, 16, 32, 64])
        self.shared_mlp_2 = SharedMLP(channels=[64, 128, 256, 512])
        self.shared_mlp_3 = SharedMLP(channels=[512, 512, 1024])

        self.shared_mlp_4 = SharedMLP(channels=[64 + 512, 256, 128, 64])

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

        score_features = torch.squeeze(score_features, dim=1)
        label_features = torch.transpose(label_features, 1, 2)

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

# Calculate Metrics in real worlds
houses = ['floor1']

data_loaders_real_word_train = {}
data_loaders_real_word_test = {}

labels = None
for house in houses:
    train, test, l, _ = get_final_doors_dataset_real_data(folder_name=house, train_size=0.25, transform_train=False)
    labels = l
    data_loader_train = DataLoader(train, batch_size=32, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
    data_loaders_real_word_train[house] = data_loader_train

    data_loader_test = DataLoader(test, batch_size=32, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
    data_loaders_real_word_test[house] = data_loader_test

yolo_gd = YOLOv5Model(model_name=YOLOv5, n_labels=len(labels.keys()), pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=globals()[f'EXP_GENERAL_DETECTOR_GIBSON_60_EPOCHS'.upper()])
yolo_gd.to('cuda')
yolo_gd.eval()

performances_in_real_worlds = {'AP': {'0': [], '1': []},
                               'TP': [], 'FP': [], 'TPm': [], 'FPiou': []}

datasets_real_worlds_train = {}
datasets_real_worlds_test = {}

with torch.no_grad():
    for house in houses:
        dataset_creator_bboxes_real_world = DatasetCreatorBBoxes()
        evaluator = MyEvaluator()
        evaluator_complete_metric = MyEvaluatorCompleteMetric()
        for images, targets, converted_boxes in tqdm(data_loaders_real_word_test[house], total=len(data_loaders_real_word_test[house]), desc=f'Evaluating yolo GD in {house}'):
            images = images.to('cuda')
            preds, train_out = yolo_gd.model(images)
            dataset_creator_bboxes_real_world.add_yolo_bboxes(images=images, targets=targets, preds=preds, bboxes_type=ExampleType.TEST)
            preds = bounding_box_filtering_yolo(preds, max_detections=300, iou_threshold=iou_threshold_matching, confidence_threshold=0.01, apply_nms=True)
            #preds = non_max_suppression(preds,0.01,0.5,multi_label=False, agnostic=True,max_det=300)
            evaluator.add_predictions_yolo(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])
            evaluator_complete_metric.add_predictions_yolo(targets=targets, predictions=preds, img_size=images.size()[2:][::-1])

        for images, targets, converted_boxes in tqdm(data_loaders_real_word_train[house], total=len(data_loaders_real_word_train[house]), desc=f'Evaluating yolo GD in {house}'):
            images = images.to('cuda')
            preds, train_out = yolo_gd.model(images)
            dataset_creator_bboxes_real_world.add_yolo_bboxes(images=images, targets=targets, preds=preds, bboxes_type=ExampleType.TRAINING)

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
        train_bboxes, test_bboxes = dataset_creator_bboxes_real_world.create_datasets(shuffle_boxes=False, apply_transforms_to_train=False)
        datasets_real_worlds_test[house] = DataLoader(test_bboxes, batch_size=32, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4, shuffle=False)
        datasets_real_worlds_train[house] = DataLoader(train_bboxes, batch_size=32, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4, shuffle=False)


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
plt.savefig('AP.svg')

fig = plt.figure()
plt.axhline(y=performances_in_real_worlds['TP'], color = 'g', linestyle = '--', label='TP')
plt.axhline(y=performances_in_real_worlds['FP'], color = 'r', linestyle = '--', label='FP')
plt.axhline(y=performances_in_real_worlds['TPm'], color = 'forestgreen', linestyle = '--', label='TPm')
plt.axhline(y=performances_in_real_worlds['FPiou'], color = 'salmon', linestyle = '--', label='FPiou')
plt.title('Complete metric')
plt.legend()
plt.savefig('complete_metric.svg')

#check_bbox_dataset(datasets_real_worlds['floor4'], confidence_threshold)
bbox_model = BboxFilterNetworkGeometric(initial_channels=7, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=TEST)
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
for data in datasets_real_worlds_train[houses[0]]:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious, target_boxes = data
    for labels in labels_encoded:
        for label in labels:
            train_total[int((label == 1).nonzero(as_tuple=True)[0])] += 1
print('Train and test total', train_total, test_total)
for data in datasets_real_worlds_test[houses[0]]:
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

    for data in tqdm(datasets_real_worlds_train[houses[0]], total=len(datasets_real_worlds_train[houses[0]]), desc=f'Training epoch {epoch}'):
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
        for data in tqdm(datasets_real_worlds_train[houses[0]], total=len(datasets_real_worlds_train[houses[0]]), desc=f'TEST epoch {epoch}'):
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

        for data in tqdm(datasets_real_worlds_test[houses[0]], total=len(datasets_real_worlds_test[houses[0]]), desc=f'TEST epoch {epoch}'):
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
    for house, dataset_real_world in datasets_real_worlds_test.items():
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
    plt.savefig('complete_metric.svg')

    print(performances_in_real_worlds_bbox_filtering)
    print(train_accuracy)
    fig = plt.figure()
    plt.plot([i for i in range(len(train_accuracy[0]))], train_accuracy[0], label='background')
    plt.plot([i for i in range(len(train_accuracy[1]))], train_accuracy[1], label='closed')
    plt.plot([i for i in range(len(train_accuracy[2]))], train_accuracy[2], label='open')
    plt.title('Train accuracy')
    plt.legend()
    plt.savefig('train_geometric.svg')

    fig = plt.figure()
    plt.plot([i for i in range(len(test_accuracy[0]))], test_accuracy[0], label='background')
    plt.plot([i for i in range(len(test_accuracy[1]))], test_accuracy[1], label='closed')
    plt.plot([i for i in range(len(test_accuracy[2]))], test_accuracy[2], label='open')
    plt.title('Val accuracy')
    plt.legend()
    plt.savefig('val_geometric.svg')
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
    plt.savefig('losses_geometric_label.svg')

    fig = plt.figure()
    plt.plot([i for i in range(len(logs['train']['loss_confidence']))], logs['train']['loss_confidence'], label='train_loss')
    plt.plot([i for i in range(len(logs['train']['loss_confidence']))], logs['test']['loss_confidence'], label='test_loss')
    plt.title('Losses')
    plt.legend()
    plt.savefig('losses_geometric_confidence.svg')
    bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})










