import os

import torch
import yaml
from torch import nn

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET
from doors_detection_long_term.doors_detector.models.generic_model import DESCRIPTION
from doors_detection_long_term.doors_detector.models.generic_model import GenericModel
from doors_detection_long_term.doors_detector.models.model_names import ModelName
from doors_detection_long_term.doors_detector.models.yolov5_repo.models.yolo import Model
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.downloads import attempt_download
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import intersect_dicts, \
    labels_to_class_weights
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.torch_utils import de_parallel
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

TEST: DESCRIPTION = 0
EXP_1_HOUSE_1_40_EPOCHS: DESCRIPTION = 1
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 2
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 3
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 4
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 5
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 6
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 7
EXP_1_HOUSE_1_60_EPOCHS: DESCRIPTION = 8
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 9
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 10
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 11
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 12
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 13
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 14
EXP_1_HOUSE_2_40_EPOCHS: DESCRIPTION = 15
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 16
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 17
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 18
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 19
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 20
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 21
EXP_1_HOUSE_2_60_EPOCHS: DESCRIPTION = 22
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 23
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 24
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 25
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 26
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 27
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 28
EXP_1_HOUSE_7_40_EPOCHS: DESCRIPTION = 29
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 30
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 31
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 32
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 33
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 34
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 35
EXP_1_HOUSE_7_60_EPOCHS: DESCRIPTION = 36
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 37
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 38
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 39
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 40
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 41
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 42
EXP_1_HOUSE_9_40_EPOCHS: DESCRIPTION = 43
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 44
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 45
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 46
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 47
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 48
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 49
EXP_1_HOUSE_9_60_EPOCHS: DESCRIPTION = 50
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 51
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 52
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 53
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 54
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 55
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 56
EXP_1_HOUSE_10_40_EPOCHS: DESCRIPTION = 57
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 58
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 59
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 60
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 61
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 62
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 63
EXP_1_HOUSE_10_60_EPOCHS: DESCRIPTION = 64
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 65
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 66
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 67
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 68
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 69
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 70
EXP_1_HOUSE_13_40_EPOCHS: DESCRIPTION = 71
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 72
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 73
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 74
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 75
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 76
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 77
EXP_1_HOUSE_13_60_EPOCHS: DESCRIPTION = 78
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 79
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 80
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 81
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 82
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 83
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 84
EXP_1_HOUSE_15_40_EPOCHS: DESCRIPTION = 85
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 86
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 87
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 88
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 89
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 90
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 91
EXP_1_HOUSE_15_60_EPOCHS: DESCRIPTION = 92
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 93
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 94
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 95
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 96
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 97
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 98
EXP_1_HOUSE_20_40_EPOCHS: DESCRIPTION = 99
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 100
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 101
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 102
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 103
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 104
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 105
EXP_1_HOUSE_20_60_EPOCHS: DESCRIPTION = 106
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 107
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 108
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 109
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 110
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 111
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 112
EXP_1_HOUSE_21_40_EPOCHS: DESCRIPTION = 113
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 114
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 115
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 116
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 117
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 118
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 119
EXP_1_HOUSE_21_60_EPOCHS: DESCRIPTION = 120
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 121
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 122
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 123
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 124
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 125
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 126
EXP_1_HOUSE_22_40_EPOCHS: DESCRIPTION = 127
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 128
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 129
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 130
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 131
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 132
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 133
EXP_1_HOUSE_22_60_EPOCHS: DESCRIPTION = 134
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 135
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 136
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 137
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 138
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 139
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 140

EXP_GENERAL_DETECTOR_GIBSON_60_EPOCHS: DESCRIPTION = 141
EXP_GENERAL_DETECTOR_DEEP_DOORS_2_60_EPOCHS: DESCRIPTION = 142

EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 143
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 144
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 145

EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 146
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 147
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 148

EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 149
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 150
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 151

EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 152
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 153
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 154

EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 155
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 156
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 157

EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 158
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 159
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 160

EXP_GENERAL_DETECTOR_GIBSON_DEEP_DOORS_2_60_EPOCHS: DESCRIPTION = 161

EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 162
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 163
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 164

EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 165
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 166
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 167

EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 168
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 169
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 170

class YOLOv5Model(GenericModel):
    def __init__(self, model_name: ModelName, n_labels: int, pretrained: bool, dataset_name: DATASET, description: DESCRIPTION):
        """

        :param model_name: the name of the detr base model
        :param n_labels: the labels' number
        :param pretrained: it refers to the DetrDoorDetector class, not to detr base model.
                            It True, the DetrDoorDetector's weights are loaded, otherwise the weights are loaded only for the detr base model
        """
        super(YOLOv5Model, self).__init__(model_name, dataset_name, description)

        with open(os.path.join(os.path.dirname(__file__), 'yolov5_repo', 'models', 'hyp.scratch-low.yaml'), errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

        self.model = Model(os.path.join(os.path.dirname(__file__), 'yolov5_repo', 'models', 'yolov5m.yaml'), ch=3, nc=n_labels, anchors=hyp.get('anchors'))

        # Set model hyper-parameters
        nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= n_labels / 80 * 3 / nl  # scale to classes and layers
        #hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = .0
        self.model.nc = n_labels
        self.nc = self.model.nc # attach number of classes to model
        self.model.class_weights = torch.Tensor()
        self.class_weights = self.model.class_weights
        self.model.names = {i: '' for i in range(n_labels)}
        self.names = self.model.names

        self.model.hyp = hyp
        self.hyp = hyp
        self.yaml = self.model.yaml

        # Load pretrained yolov5
        if not pretrained:
            state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'yolov5_repo', 'models', 'yolov7m_state_dict.pth'))
            csd = intersect_dicts(state_dict, self.model.state_dict(), exclude=[])  # intersect
            self.model.load_state_dict(csd, strict=False)

        else:
            path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            state_dict = torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu'))
            csd = intersect_dicts(state_dict, self.model.state_dict(), exclude=[])  # intersect
            self.model.load_state_dict(csd)

    def forward(self, x):

        x = self.model(x)
        return x

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model.to(device)