import cv2
import torch.optim
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.dataset_bboxes.DatasetCreatorBBoxes import DatasetsCreatorBBoxes, \
    ExampleType
from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.bbox_filter_network import SharedMLP
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5, BBOX_FILTER_NETWORK_GEOMETRIC
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_yolov5, collate_fn_bboxes
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *

colors = {0: (0, 0, 255), 1: (0, 255, 0)}
num_bboxes = 20

iou_threshold_matching = 0.5

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

        score_features = torch.squeeze(score_features)
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
        labels_loss = torch.log(labels_features) * label_targets * torch.tensor([[0.3, 0.5, 0.2]], device=label_targets.device)
        labels_loss = torch.mean(torch.sum(torch.sum(labels_loss, 2) * -1, 1))

        return torch.tensor(0), labels_loss

dataset_creator_bboxes = DatasetsCreatorBBoxes()
dataset_creator_bboxes.load_dataset(folder_name='yolov5_simulation_dataset')
dataset_creator_bboxes.select_n_bounding_boxes(num_bboxes=num_bboxes)
dataset_creator_bboxes.match_bboxes_with_gt(iou_threshold_matching=iou_threshold_matching)

train_bboxes, test_bboxes = dataset_creator_bboxes.create_datasets(shuffle_boxes=False)

train_dataset_bboxes = DataLoader(train_bboxes, batch_size=64, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4, shuffle=False)
test_dataset_bboxes = DataLoader(test_bboxes, batch_size=64, collate_fn=collate_fn_bboxes(use_confidence=True), num_workers=4)

#Check the dataset
def check_bbox_dataset(dataset):
    for data in dataset:
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
        detected_bboxes = torch.transpose(detected_bboxes, 1, 2)
        images_opencv = []
        w_image, h_image = images.size()[2:][::-1]
        for image, detected_list, fixed_list, confidences_list, labels_list, ious_list in zip(images, detected_bboxes.tolist(), fixed_bboxes.tolist(), confidences.tolist(), labels_encoded.tolist(), ious.tolist()):
            image = image.to('cpu')
            image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            image_detected = cv2.cvtColor(np.transpose(np.array(image), (1, 2, 0)), cv2.COLOR_RGB2BGR)
            image_detected_high_conf = image_detected.copy()
            image_matched = image_detected.copy()
            for (cx, cy, w, h, confidence, closed, open), (cx_f, cy_f, w_f, h_f) in zip(detected_list, fixed_list):
                #print(cx, cy, w, h)
                x, y, x2, y2 = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) * np.array([w_image, h_image, w_image, h_image])
                x, y, x2, y2 = round(x), round(y), round(x2), round(y2)
                print(x, y, x2, y2)
                label = 0 if closed == 1 else 1
                image_detected = cv2.rectangle(image_detected, (x, y),
                                                       (x2, y2), colors[label], 2)
                if confidence >= 0.75:
                    image_detected_high_conf = cv2.rectangle(image_detected_high_conf, (x, y),
                                                   (x2, y2), colors[label], 2)

                if not(cx == cx_f and cy == cy_f and w == w_f and h == h_f):
                    image_matched = cv2.rectangle(image_matched, (x, y),
                                                   (x2, y2), colors[label], 2)
                else:
                    image_matched = cv2.rectangle(image_matched, (x, y),
                                                  (x2, y2),(0,0,0), 2)

            images_opencv.append(cv2.hconcat([image_detected, image_detected_high_conf, image_matched]))
        new_image = cv2.vconcat(images_opencv)
        cv2.imshow('show', new_image)
        cv2.waitKey()

#check_bbox_dataset(train_dataset_bboxes)
bbox_model = BboxFilterNetworkGeometric(initial_channels=7, n_labels=3, model_name=BBOX_FILTER_NETWORK_GEOMETRIC, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=TEST)
bbox_model.to('cuda')

criterion = BboxFilterNetworkGeometricLoss(reduction_image='sum', reduction_global='mean')

optimizer = optim.Adam(bbox_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion.to('cuda')
#for n, p in bbox_model.named_parameters():
#    if p.requires_grad:
#        print(n)

logs = {'train': [], 'test': [], 'ap': {0: [], 1: []}, 'complete_metric': {'TP': [], 'FP': [], 'BFD': []}}

# compute total labels
train_total = {0:0, 1:0, 2:0}
test_total = {0:0, 1:0, 2:0}
for data in train_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
    for labels in labels_encoded:
        for label in labels:
            train_total[int((label == 1).nonzero(as_tuple=True)[0])] += 1

for data in test_dataset_bboxes:
    images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
    for labels in labels_encoded:
        for label in labels:
            test_total[int((label == 1).nonzero(as_tuple=True)[0])] += 1

train_accuracy = {0: [], 1: [], 2: []}
test_accuracy = {0: [], 1: [], 2: []}

for epoch in range(60):
    #scheduler.step()
    bbox_model.train()
    criterion.train()
    optimizer.zero_grad()

    temp_losses = []
    for data in tqdm(train_dataset_bboxes, total=len(train_dataset_bboxes), desc=f'Training epoch {epoch}'):
        images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
        images = images.to('cuda')
        detected_bboxes = detected_bboxes.to('cuda')
        confidences = confidences.to('cuda')
        labels_encoded = labels_encoded.to('cuda')

        preds = bbox_model(images, detected_bboxes)
        #print(preds, filtered)
        loss_confidence, loss_label = criterion(preds, confidences, labels_encoded)
        final_loss = loss_label

        temp_losses.append(final_loss.item())
        optimizer.zero_grad()
        final_loss.backward()
        #official_loss.backward()
        optimizer.step()
    logs['train'].append(sum(temp_losses) / len(temp_losses))

    temp_losses = []

    with torch.no_grad():

        bbox_model.eval()
        criterion.eval()

        temp_accuracy = {0:0, 1:0, 2:0}
        for data in tqdm(train_dataset_bboxes, total=len(train_dataset_bboxes), desc=f'TEST epoch {epoch}'):
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
            images = images.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')

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
            images, detected_bboxes, fixed_bboxes, confidences, labels_encoded, ious = data
            images = images.to('cuda')
            detected_bboxes = detected_bboxes.to('cuda')
            confidences = confidences.to('cuda')
            labels_encoded = labels_encoded.to('cuda')

            preds = bbox_model(images, detected_bboxes)
            loss_confidence, loss_label = criterion(preds, confidences, labels_encoded)
            final_loss = loss_label

            temp_losses.append(final_loss.item())

            predicted_labels = preds[1]

            for predicted, target in zip(predicted_labels, labels_encoded):
                values, p_indexes = torch.max(predicted, dim=1)
                values, gt_indexes = torch.max(target, dim=1)
                for p_index, gt_index in zip(p_indexes.tolist(), gt_indexes.tolist()):
                    if gt_index == p_index:
                        temp_accuracy[int(gt_index)] += 1

        for i in range(3):
            test_accuracy[i].append(temp_accuracy[i] / test_total[i])

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
    logs['test'].append(sum(temp_losses) / len(temp_losses))
    print(logs['train'], logs['test'])

    fig = plt.figure()
    plt.plot([i for i in range(len(logs['train']))], logs['train'], label='train_loss')
    plt.plot([i for i in range(len(logs['train']))], logs['test'], label='test_loss')
    plt.title('Losses')
    plt.legend()
    plt.savefig('losses_geometric.svg')
    bbox_model.save(epoch=epoch, optimizer_state_dict=optimizer.state_dict(), params={}, logs=logs, lr_scheduler_state_dict={})







