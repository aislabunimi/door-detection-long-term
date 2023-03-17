import random
import time

from torch.optim import lr_scheduler
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
import numpy as np
import torch.optim
from torch.utils.data import DataLoader
from doors_detection_long_term.doors_detector.dataset.dataset_doors_final.datasets_creator_doors_final import DatasetsCreatorDoorsFinal
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.models.faster_rcnn import *

from doors_detection_long_term.doors_detector.utilities.plot import plot_losses
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything


device = 'cuda'

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']
epochs_general_detector = [40, 60]
epochs_qualified_detectors = [20, 40]
fine_tune_quantity = [15, 25, 50, 75]


# Params
params = {
    #'epochs': 40,
    'batch_size': 2,
    'seed': 0
}
def prepare_model(description, reload_model, restart_checkpoint):
    model = FasterRCNN(model_name=FASTER_RCNN, n_labels=len(labels.keys()), pretrained=reload_model, dataset_name=FINAL_DOORS_DATASET, description=description)

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

    optimizer = torch.optim.SGD([p for n, p in params], lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

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
    for house in houses:
        epoch_count = 0
        train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name=house.replace('_', ''), train_size=0.25, use_negatives=False)
        print(f'Train set size: {len(train)}', f'Validation set size: {len(validation)}', f'Test set size: {len(test)}')
        data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)
        data_loader_validation = DataLoader(validation, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)
        data_loader_test = DataLoader(test, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

        model, optimizer, scheduler, logs = prepare_model(globals()[f'EXP_1_{house}_{epochs_general_detector[0]}_EPOCHS'.upper()], reload_model=False, restart_checkpoint=False)
        print_logs_every = 10
        model.to('cuda')

        start_time = time.time()

        for epoch in range(epochs_general_detector[-1]):

            temp_logs = {'train': [], 'train_after_backpropagation': [], 'validation': [], 'test': []}
            accumulate_loss = []

            model.train()
            optimizer.zero_grad()
            for d, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=f'{house} - Epoch {epoch} - Train model'):
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
                scheduler.step()

                accumulate_loss.append(loss_value)

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
                for i, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=f'{house} - Epoch {epoch} - Test model with training data'):
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
                for i, data in tqdm(enumerate(data_loader_validation), total=len(data_loader_validation), desc=f'{house} - Epoch {epoch} - Test model with validation data'):
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

            # Test
            with torch.no_grad():
                accumulate_loss = []
                for i, data in tqdm(enumerate(data_loader_test), total=len(data_loader_test), desc=f'{house} - Epoch {epoch} - Test model with test data'):
                    images, targets, new_targets = data
                    images = list(image.to(device) for image in images)
                    new_targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in new_targets]


                    with torch.cuda.amp.autocast(enabled=False):
                        loss_dict = model(images, new_targets)
                        losses = sum(loss for loss in loss_dict.values())
                        print('losses', loss_dict)
                        losses_reduced = sum(loss for loss in loss_dict.values())

                        loss_value = losses_reduced.item()
                        accumulate_loss.append(loss_value)

            logs['test'].append({'loss': sum(accumulate_loss, 0) / len(accumulate_loss)})
            print(f'----> EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch]}' for k, v in logs.items()]))

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
                model.set_description(globals()[f'EXP_1_{house}_{epochs_general_detector[epoch_count]}_EPOCHS'.upper()])


    # Qualify the general detectors trained before
    for house, epochs_general, quantity in [(h, e, q) for h in houses for e in epochs_general_detector for q in fine_tune_quantity]:
        epoch_count = 0
        print(f'{house}, general detectors trained for {epochs_general} epochs, fine tune train set: {quantity}')
        train, validation, test, labels, _ = get_final_doors_dataset_epoch_analysis(experiment=2, folder_name=house.replace('_', ''), train_size=quantity / 100, use_negatives=False)
        print(f'Train set size: {len(train)}', f'Validation set size: {len(validation)}', f'Test set size: {len(test)}')
        data_loader_train = DataLoader(train, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, shuffle=False, num_workers=4)
        data_loader_validation = DataLoader(validation, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)
        data_loader_test = DataLoader(test, batch_size=params['batch_size'], collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

        model, optimizer, scheduler, logs = prepare_model(globals()[f'EXP_1_{house}_{epochs_general_detector[0]}_EPOCHS'.upper()], reload_model=False, restart_checkpoint=False)
        print_logs_every = 10

        model.set_description(globals()[f'EXP_2_{house}_EPOCHS_GD_{epochs_general}_EPOCH_QD_{epochs_qualified_detectors[0]}_fine_tune_{quantity}'.upper()])

        model.to('cuda')
        start_time = time.time()

        for epoch in range(epochs_qualified_detectors[-1]):

            temp_logs = {'train': [], 'train_after_backpropagation': [], 'validation': [], 'test': []}
            accumulate_loss = []

            model.train()
            optimizer.zero_grad()
            for d, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=f'{house} - Epoch {epoch} - Fine tune with {quantity}% of examples'):
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
                scheduler.step()

                accumulate_loss.append(loss_value)

            logs['time'].append(time.time() - start_time)
            epoch_total = {}
            for d in temp_logs['train']:
                for k in d:
                    epoch_total[k] = epoch_total.get(k, 0) + d[k]

            logs['train'].append({'loss': sum(accumulate_loss, 0) / len(accumulate_loss)})

            # Train loss after backpropagation
            with torch.no_grad():
                accumulate_loss = []
                for i, data in tqdm(enumerate(data_loader_train), total=len(data_loader_train), desc=f'{house} - Epoch {epoch} - Test model with training data'):
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
                for i, data in tqdm(enumerate(data_loader_validation), total=len(data_loader_validation), desc=f'{house} - Epoch {epoch} - Test model with validation data'):
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

            # Test
            with torch.no_grad():
                accumulate_loss = []
                for i, data in tqdm(enumerate(data_loader_test), total=len(data_loader_test), desc=f'{house} - Epoch {epoch} - Test model with test data'):
                    images, targets, new_targets = data
                    images = list(image.to(device) for image in images)
                    new_targets = [{k: v.to(device).to(torch.int64) for k, v in t.items()} for t in new_targets]


                    with torch.cuda.amp.autocast(enabled=False):
                        loss_dict = model(images, new_targets)
                        losses = sum(loss for loss in loss_dict.values())
                        #print('losses', loss_dict)

                    loss_value = losses.item()
                    accumulate_loss.append(loss_value)

            logs['test'].append({'loss': sum(accumulate_loss, 0) / len(accumulate_loss)})
            print(f'----> EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch]}' for k, v in logs.items()]))

            plot_losses(logs)

            model.save(epoch=epoch,
                       optimizer_state_dict=optimizer.state_dict(),
                       lr_scheduler_state_dict=scheduler.state_dict(),
                       params=params,
                       logs=logs,
                       )
            # Change the model description on each epoch step
            if epoch == epochs_qualified_detectors[epoch_count] - 1 and epoch_count < len(epochs_qualified_detectors) -1:
                epoch_count += 1
                model.set_description(globals()[f'EXP_2_{house}_EPOCHS_GD_{epochs_general}_EPOCH_QD_{epochs_qualified_detectors[epoch_count]}_fine_tune_{quantity}'.upper()])