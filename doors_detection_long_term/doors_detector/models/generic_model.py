import os

import torch
from torch import nn

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET
from doors_detection_long_term.doors_detector.models.model_names import ModelName
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

DESCRIPTION = int

class GenericModel(nn.Module):
    def __init__(self, model_name: ModelName, dataset_name: DATASET, description: DESCRIPTION):
        super(GenericModel, self).__init__()

        self._model_name = model_name
        self._dataset_name = dataset_name
        self._description = description

    def save(self, epoch, optimizer_state_dict, lr_scheduler_state_dict, params, logs):
        path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
        if trained_models_path == "":
            path = os.path.join(os.path.dirname(__file__), path)
        else:
            path = os.path.join(trained_models_path, path)

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        torch.save(
            {
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_scheduler_state_dict
            }, os.path.join(path, 'checkpoint.pth')
        )

        torch.save(
            {
                'epoch': epoch,
                'logs': logs,
                'params': params,
            }, os.path.join(path, 'training_data.pth')
        )

    def load_checkpoint(self,):
        path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
        if trained_models_path == "":
            path = os.path.join(os.path.dirname(__file__), path)
        else:
            path = os.path.join(trained_models_path, path)

        checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
        training_data = torch.load(os.path.join(path, 'training_data.pth'))

        return {**checkpoint, **training_data}

    def set_description(self, description: DESCRIPTION):
        self._description = description
