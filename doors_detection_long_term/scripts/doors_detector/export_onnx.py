import onnx
import torch.onnx

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50, YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5 import *
from doors_detection_long_term.doors_detector.utilities.utils import seed_everything
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_epoch_analysis

device = 'cuda'
batch_size = 1

if __name__ == '__main__':
    seed_everything(0)

    #train, test, labels, COLORS = get_final_doors_dataset(experiment=2, folder_name='house1', train_size=0.25, use_negatives=True)
    #train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
    train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, folder_name='house1', train_size=0.25, use_negatives=True)


    print(f'Train set size: {len(train)}', f'Test set size: {len(test)}')

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=EXP_GENERAL_DETECTOR_2_LAYERS_BACKBONE_GIBSON_60_EPOCHS)
    model_2 = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50)

    model.eval()
    model_2.train()


    #torch.onnx.export(model.model, torch.ones(1, 3, 240, 320), 'detr_onnx.onnx',
     #                 export_params=True,opset_version=12)

    torch.onnx.export(model_2.model, torch.ones(1, 3, 320, 320), 'yolov5_onnx.onnx',
                      export_params=True,opset_version=11)

    model_onnx = onnx.load('yolov5_onnx.onnx')
    onnx.checker.check_model(model_onnx)