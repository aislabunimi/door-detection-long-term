import time

import numpy as np
import onnx
import onnxruntime
import torch

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET

from doors_detection_long_term.doors_detector.models.detr_door_detector import \
    EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_15_EPOCHS_40, DetrDoorDetector
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    print(onnxruntime.get_device())

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_15_EPOCHS_40)
    model.eval()
    model_path = 'model_onnx.onnx'

    providers = [('CUDAExecutionProvider', {
        'device_id': 0,
        'cudnn_conv_algo_search': 'DEFAULT',
    })]
    input_tensor = torch.ones(1, 3, 320, 320)
    torch.onnx.export(model.model, input_tensor, model_path, input_names=['input'],
                      output_names=['output'], export_params=True, do_constant_folding=True)

    onnx_model = onnx.load("model_onnx.onnx")
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("model_onnx.onnx", providers=providers)



    # compute ONNX Runtime output prediction
    times = []
    for i in range(200):
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input_tensor)}
        t = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        times.append(time.time() - t)
        print(ort_outs)
    print(1/(sum(times) / len(times)))
    torch_out = model.model(input_tensor)
    torch_out = [torch_out['pred_logits'], torch_out['pred_boxes']]
    #np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outs[1], rtol=1e-03, atol=1e-05)




