import random
import time

from torch.optim import lr_scheduler
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET, \
    IGIBSON_DATASET
import numpy as np
import torch.optim
from torch.utils.data import DataLoader
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.models.faster_rcnn import *

from doors_detection_long_term.doors_detector.utilities.plot import plot_losses
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import seed_everything


device = 'cuda'

epochs_general_detector = [20, 40, 60]


# Params
params = {
    #'epochs': 40,
    'batch_size': 4,
    'seed': 0
}
def prepare_model(description, reload_model, restart_checkpoint):
    model = FasterRCNN(model_name=FASTER_RCNN, n_labels=len(labels.keys()) + 1, pretrained=reload_model, dataset_name=IGIBSON_DATASET, description=description)

    logs = {'train': [], 'train_after_backpropagation': [], 'validation': [], 'test': [], 'time': []}
    optimizer_state_dict = {}
    lr_scheduler_state_dict = {}
    start_epoch = 0
    if restart_checkpoint:
        checkpoint = model.load_checkpoint()
        start_epoch = checkpoint['epoch'] + 1
        logs = checkpoint['logs']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        lr_scheduler_state_dict = checkpoint['lr_scheduler_state_dict']


    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    optimizer = torch.optim.SGD([p for n, p in params], lr=0.001,
                                momentum=0.9, nesterov=True)

    if restart_checkpoint:
        optimizer.load_state_dict(optimizer_state_dict)

    optimizer.zero_grad()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    if restart_checkpoint:
        scheduler.load_state_dict(lr_scheduler_state_dict)

    return model, optimizer, scheduler, logs


if __name__ == '__main__':

    # Fix seeds
    seed_everything(params['seed'])

    # Train the general detector with multiple epochs

    epoch_count = 0
    train, validation, labels, _ = get_igibson_dataset_all_scenes(doors_config='realistic', step=None)

    print(f'Train set size: {len(train)}', f'Validation set size: {len(validation)}', f'Test set size: {len(validation)}')
    data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)
    data_loader_validation = DataLoader(validation, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

    model, optimizer, scheduler, logs = prepare_model(globals()[f'EXP_GENERAL_DETECTOR_IGIBSON_{epochs_general_detector[0]}_EPOCHS'.upper()], reload_model=False, restart_checkpoint=False)
    print_logs_every = 1000
    model.to('cuda')

    start_time = time.time()

    for epoch in range(epochs_general_detector[-1]):

        temp_logs = {'train': [], 'train_after_backpropagation': [], 'validation': [], 'test': []}
        accumulate_loss = []

        model.train()
        optimizer.zero_grad()
        for d, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=f'Training - Epoch {epoch} - Train model'):
            images, targets, new_targets = data
            images = list(image.to(device) for image in images)
            new_targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in new_targets]


            with torch.cuda.amp.autocast(enabled=False):
                loss_dict = model(images, new_targets)
                losses = sum(loss for loss in loss_dict.values())
                #print('losses', loss_dict)

                loss_value = losses.item()
                #print('loss_value', loss_value)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                accumulate_loss.append(loss_value)

            #scheduler.step()
            logs['time'].append(time.time() - start_time)
            epoch_total = {}
            for d in temp_logs['train']:
                for k in d:
                    epoch_total[k] = epoch_total.get(k, 0) + d[k]

            logs['train'].append({'loss': sum(accumulate_loss, 0) / len(accumulate_loss)})

            #print(f'----> {house} - EPOCH SUMMARY TRAIN [{epoch}] -> ' + ', '.join([f'{k}: {v}' for k, v in logs['train'][epoch].items()]))

        # Train loss after backpropagation
        with torch.no_grad():
            accumulate_loss = []
            for i, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=f'Epoch {epoch} - Test model with training data'):
                images, targets, new_targets = data
                images = list(image.to(device) for image in images)
                new_targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in new_targets]


                with torch.cuda.amp.autocast(enabled=False):
                    loss_dict = model(images, new_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    #print('losses', loss_dict)

                loss_value = losses.item()
                accumulate_loss.append(loss_value)

        logs['train_after_backpropagation'].append({'loss': sum(accumulate_loss, 0) / len(accumulate_loss)})

        #print(f'----> EPOCH SUMMARY TRAIN AFTER BACKPROP [{epoch}] -> [{i}/{len(data_loader_train)}]: ' + ', '.join([f'{k}: {v}' for k, v in logs['train_after_backpropagation'][epoch].items()]))

        # Validation
        with torch.no_grad():
            accumulate_loss = []
            for i, data in tqdm(enumerate(data_loader_validation), total=len(data_loader_validation), desc=f'Epoch {epoch} - Test model with validation data'):
                images, targets, new_targets = data
                images = list(image.to(device) for image in images)
                new_targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in new_targets]


                with torch.cuda.amp.autocast(enabled=False):
                    loss_dict = model(images, new_targets)
                    losses = sum(loss for loss in loss_dict.values())
                    #print('losses', loss_dict)

                loss_value = losses.item()
                accumulate_loss.append(loss_value)

        logs['validation'].append({'loss': sum(accumulate_loss, 0) / len(accumulate_loss)})

        plot_losses(logs)

        model.save(epoch=epoch,
                   optimizer_state_dict=optimizer.state_dict(),
                   lr_scheduler_state_dict=scheduler.state_dict(),
                   params=params,
                   logs=logs,
                   )

        # Change the model description on each epoch step
        if epoch == epochs_general_detector[epoch_count] - 1 and epoch_count < len(epochs_general_detector) -1:
            epoch_count += 1
            model.set_description(globals()[f'EXP_GENERAL_DETECTOR_IGIBSON_{epochs_general_detector[epoch_count]}_EPOCHS'.upper()])