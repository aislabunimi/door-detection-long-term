import pycuda.autoinit
import numpy as np
import onnx
import tensorrt
import torch
import tensorrt as trt
from torch.utils.data import DataLoader
from tqdm import tqdm
import pycuda.driver as cuda

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.evaluators.my_evaluator import MyEvaluator
from doors_detection_long_term.doors_detector.evaluators.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from doors_detection_long_term.doors_detector.models.detr_door_detector import \
    EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_15_EPOCHS_40, DetrDoorDetector
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.utils import collate_fn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_real_data
from doors_detection_long_term.scripts.doors_detector.tensorrt.tensorrt_converter import model_to_tensorrt


if __name__ == '__main__':
    model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_60_FINE_TUNE_15_EPOCHS_40)
    engine, context = model_to_tensorrt(model.model, input_tensor=torch.zeros(1, 3, 240, 320))

    device_input_buffer = None
    host_output_buffers = []
    device_output_buffers = []

    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:  # we expect only one input
            input_shape = engine.get_tensor_shape(binding)
            print('INPUT', input_shape)
            host_input_buffer = np.zeros(input_shape, dtype=np.float32)
            input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
            device_input_buffer = cuda.mem_alloc(input_size)

        elif engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:  # and one output
            output_shape = engine.get_tensor_shape(binding)
            print('OUTPUT', output_shape)
            host_output_buffers.append(np.zeros(output_shape, dtype=np.float32))
            host_output_pagelocked = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            device_output_buffers.append(cuda.mem_alloc(host_output_pagelocked.nbytes))

    _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn, drop_last=False, num_workers=4)


    evaluator = MyEvaluator()
    evaluator_complete_metric = MyEvaluatorCompleteMetric()


    stream = cuda.Stream()
    for images, targets in tqdm(data_loader_test, total=len(data_loader_test), desc='Inference Tensorrt'):


        cuda.memcpy_htod_async(device_input_buffer, images.numpy(), stream)

        context.execute_async_v2([int(device_input_buffer)] + [int(i) for i in device_output_buffers], stream.handle, None)

        for host_buffer, device_buffer in zip(host_output_buffers, device_output_buffers):
            cuda.memcpy_dtoh_async(host_buffer, device_buffer, stream)

        stream.synchronize()

        outputs = {'pred_logits': torch.from_numpy(host_output_buffers[0]),
                   'pred_boxes': torch.from_numpy(host_output_buffers[1])}

        evaluator.add_predictions(targets=targets, predictions=outputs)
        evaluator_complete_metric.add_predictions(targets=targets, predictions=outputs)

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)
    complete_metrics = evaluator_complete_metric.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)

    mAP = 0
    print('Results per bounding box:')
    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        mAP += values['AP']
        print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
        #print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
    #print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')
