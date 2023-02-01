import random
import time


from matplotlib import pyplot as plt
from torch.optim import lr_scheduler

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
import numpy as np
import torch.optim
from torch.utils.data import DataLoader
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.torch_utils import smart_optimizer, de_parallel
from doors_detection_long_term.doors_detector.utilities.plot import plot_losses
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn
from dataset_configurator import *
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.loss import ComputeLoss
import torchvision.transforms as T


device = 'cuda'

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']
epochs_qualified_detectors = [20, 40]
fine_tune_quantity = [25, 50, 75]


# Params
params = {
    'batch_size': 1,
    'epochs': 40
}

train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.25, use_negatives=False)
print(f'Train set size: {len(train)}', f'Validation set size: {len(validation)}', f'Test set size: {len(test)}')
data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=False, num_workers=4)
data_loader_validation = DataLoader(validation, batch_size=params['batch_size'], collate_fn=collate_fn, drop_last=False, num_workers=4)
data_loader_test = DataLoader(test, batch_size=params['batch_size'], collate_fn=collate_fn, drop_last=False, num_workers=4)
model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_1)

# Model parameters
nl = de_parallel(model.model).model[-1].nl

model.train()
model.to('cuda')
compute_loss = ComputeLoss(model.model)

# General paramaters
nb = len(data_loader_train)
nw = max(round(model.hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
nbs = 64
accumulate = max(round(nbs / params['batch_size']), 1)
model.hyp['weight_decay'] *= params['batch_size'] * accumulate / nbs
model.model.hyp['weight_decay'] *= params['batch_size'] * accumulate / nbs

# Optimizer
optimizer = smart_optimizer(model.model, 'SGD', model.hyp['lr0'], model.hyp['momentum'], model.hyp['weight_decay'])

# Scheduler
lf = lambda x: (1 - x / params['epochs']) * (1.0 - model.hyp['lrf']) + model.hyp['lrf']  # linear
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)



logs = {'train': [], 'train_after_backpropagation': [], 'validation': [], 'test': []}
for epoch in range(params['epochs']):
    temp_logs = {'train': [], 'train_after_backpropagation': [], 'validation': [], 'test': []}
    model.train()

    for d, data in enumerate(data_loader_train):

        ni = d + nb * epoch

        # warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            accumulate = max(1, np.interp(ni, xi, [1, nbs / params['batch_size']]).round())
            for j, x in enumerate(optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [model.hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [model.hyp['warmup_momentum'], model.hyp['momentum']])

        images, targets = data
        batch_size_width, batch_size_height = images.size()[2], images.size()[3]
        converted_boxes = []
        for i, target in enumerate(targets):
            real_size_width, real_size_height = target['size'][0], target['size'][1]
            scale_boxes = torch.tensor([[real_size_width / batch_size_width, real_size_height / batch_size_height, real_size_width / batch_size_width, real_size_height / batch_size_height]])
            converted_boxes.append(torch.cat([
                torch.tensor([[i] for _ in range(int(list(target['labels'].size())[0]))]),
                torch.reshape(target['labels'], (target['labels'].size()[0], 1)),
                target['boxes'] * scale_boxes
                ], dim=1))
            #print('PRINT?', target['size'], images.size()[2:], target['boxes'], target['boxes'] * scale_boxes)
        converted_boxes = torch.cat(converted_boxes, dim=0)


        images = images.to('cuda')
        output = model(images)
        loss, loss_items = compute_loss(output, converted_boxes.to('cuda'))
        #print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp_logs['train'].append(loss.item())
        #print(temp_logs)
        if d % 10 == 0:
            print(f'EPOCHS {epoch}, [{d}:{len(data_loader_train)}]')

    logs['train'].append(sum(temp_logs['train']) / len(temp_logs['train']))

    print(logs['train'])




