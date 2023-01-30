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
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import intersect_dicts
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

EXP_1_HOUSE_1: DESCRIPTION = 1
EXP_2_HOUSE_1_FINE_TUNE_25: DESCRIPTION = 2
EXP_2_HOUSE_1_FINE_TUNE_50: DESCRIPTION = 3
EXP_2_HOUSE_1_FINE_TUNE_75: DESCRIPTION = 4
EXP_1_HOUSE_2: DESCRIPTION = 5
EXP_2_HOUSE_2_FINE_TUNE_25: DESCRIPTION = 6
EXP_2_HOUSE_2_FINE_TUNE_50: DESCRIPTION = 7
EXP_2_HOUSE_2_FINE_TUNE_75: DESCRIPTION = 8
EXP_1_HOUSE_7: DESCRIPTION = 9
EXP_2_HOUSE_7_FINE_TUNE_25: DESCRIPTION = 10
EXP_2_HOUSE_7_FINE_TUNE_50: DESCRIPTION = 11
EXP_2_HOUSE_7_FINE_TUNE_75: DESCRIPTION = 12
EXP_1_HOUSE_9: DESCRIPTION = 13
EXP_2_HOUSE_9_FINE_TUNE_25: DESCRIPTION = 14
EXP_2_HOUSE_9_FINE_TUNE_50: DESCRIPTION = 15
EXP_2_HOUSE_9_FINE_TUNE_75: DESCRIPTION = 16
EXP_1_HOUSE_10: DESCRIPTION = 17
EXP_2_HOUSE_10_FINE_TUNE_25: DESCRIPTION = 18
EXP_2_HOUSE_10_FINE_TUNE_50: DESCRIPTION = 19
EXP_2_HOUSE_10_FINE_TUNE_75: DESCRIPTION = 20
EXP_1_HOUSE_13: DESCRIPTION = 21
EXP_2_HOUSE_13_FINE_TUNE_25: DESCRIPTION = 22
EXP_2_HOUSE_13_FINE_TUNE_50: DESCRIPTION = 23
EXP_2_HOUSE_13_FINE_TUNE_75: DESCRIPTION = 24
EXP_1_HOUSE_15: DESCRIPTION = 25
EXP_2_HOUSE_15_FINE_TUNE_25: DESCRIPTION = 26
EXP_2_HOUSE_15_FINE_TUNE_50: DESCRIPTION = 27
EXP_2_HOUSE_15_FINE_TUNE_75: DESCRIPTION = 28
EXP_1_HOUSE_20: DESCRIPTION = 29
EXP_2_HOUSE_20_FINE_TUNE_25: DESCRIPTION = 30
EXP_2_HOUSE_20_FINE_TUNE_50: DESCRIPTION = 31
EXP_2_HOUSE_20_FINE_TUNE_75: DESCRIPTION = 32
EXP_1_HOUSE_21: DESCRIPTION = 33
EXP_2_HOUSE_21_FINE_TUNE_25: DESCRIPTION = 34
EXP_2_HOUSE_21_FINE_TUNE_50: DESCRIPTION = 35
EXP_2_HOUSE_21_FINE_TUNE_75: DESCRIPTION = 36
EXP_1_HOUSE_22: DESCRIPTION = 37
EXP_2_HOUSE_22_FINE_TUNE_25: DESCRIPTION = 38
EXP_2_HOUSE_22_FINE_TUNE_50: DESCRIPTION = 39
EXP_2_HOUSE_22_FINE_TUNE_75: DESCRIPTION = 40

class YOLOv5Model(GenericModel):
    def __init__(self, model_name: ModelName, n_labels: int, pretrained: bool, dataset_name: DATASET, description: DESCRIPTION):
        """

        :param model_name: the name of the detr base model
        :param n_labels: the labels' number
        :param pretrained: it refers to the DetrDoorDetector class, not to detr base model.
                            It True, the DetrDoorDetector's weights are loaded, otherwise the weights are loaded only for the detr base model
        """
        super(YOLOv5Model, self).__init__(model_name, dataset_name, description)
        #weights = attempt_download(os.path.join(os.path.dirname(__file__), 'yolov5_repo', 'models', 'yolov5m.pt'))  # download if not found locally
        #print('WEIGHTSSSS', weights)
        #ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        #torch.save(ckpt['model'].state_dict(), os.path.join(os.path.dirname(__file__), 'yolov7m_state_dict.pth'))
        with open(os.path.join(os.path.dirname(__file__), 'yolov5_repo', 'models', 'hyp.scratch-low.yaml'), errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict

        self.model = Model(os.path.join(os.path.dirname(__file__), 'yolov5_repo', 'models', 'yolov5m.yaml'), ch=3, nc=n_labels, anchors=hyp.get('anchors'))
        self.model.hyp = hyp
        self.hyp = hyp

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
            self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, x):

        x = self.model(x)

        """
        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape=[batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the un-normalized bounding box.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        return x

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model.to(device)