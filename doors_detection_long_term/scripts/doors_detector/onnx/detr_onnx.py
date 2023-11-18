import time

import numpy as np
import onnx
import onnxruntime
import torch
from onnxconverter_common import float16

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET

from doors_detection_long_term.doors_detector.models.detr_door_detector import \
    EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_15_EPOCHS_40, DetrDoorDetector
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50

def to_numpy(tensor):
    return tensor.detach().cpu().numpy().astype(np.float16) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float16)

if __name__ == '__main__':
    print(onnxruntime.get_device())

    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=False, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_15_EPOCHS_40)
    model.eval()
    model_path = 'model_onnx.onnx'

    providers = [('CUDAExecutionProvider', {
        'device_id': 0,
        'cudnn_conv_algo_search': 'DEFAULT',
    })]
    input_tensor = torch.ones(1, 3, 240, 320)
    torch.onnx.export(model.model, input_tensor, model_path, input_names=['input'],
                      output_names=['output'], export_params=True, do_constant_folding=True)

    onnx_model = onnx.load("model_onnx.onnx")
    onnx_model = float16.convert_float_to_float16(onnx_model)
    onnx.save(onnx_model, "model_onnx.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession("model_onnx.onnx", providers=providers)

    io_binding = ort_session.io_binding()
    io_binding.bind_cpu_input('input', to_numpy(input_tensor))
    io_binding.bind_output('output')

    # compute ONNX Runtime output prediction
    times = []
    for i in range(200):
        t = time.time()
        ort_outs = ort_session.run_with_iobinding(io_binding)
        times.append(time.time() - t)
    print(f'FPS: {1/(sum(times) / len(times))}')





