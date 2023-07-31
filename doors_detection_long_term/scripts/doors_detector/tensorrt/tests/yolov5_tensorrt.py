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
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5 import YOLOv5Model, \
    EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15
from doors_detection_long_term.doors_detector.models.yolov5_repo.utils.general import non_max_suppression
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import get_final_doors_dataset_real_data


def yolov5_to_tensorrt(model: YOLOv5Model, input_tensor: torch.Tensor=torch.ones(1, 3, 320, 320), fp16: bool=False):

    model.eval()

    # Export to ONNX format
    model_path = 'model_onnx.onnx'

    torch.onnx.export(model.model, input_tensor, model_path, input_names=['input'],
                      output_names=['output'], export_params=True)
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)

    # Import ONNX to tensorrt
    TRT_LOGGER = trt.Logger(min_severity=tensorrt.Logger.WARNING)

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse ONNX
    success = parser.parse_from_file(model_path)

    if parser.num_errors > 0:
        print('Parsing model failed with the following errors:')
    for idx in range(parser.num_errors):
        print('\t' + parser.get_error(idx))

    if not success:
        raise Exception('Parsing onnx model failed')

    config = builder.create_builder_config()

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print('------------------- Convert to FP16')

    print('Building an engine...')
    serialized_engine = builder.build_serialized_network(network, config)
    print("Completed creating Engine")

    runtime = trt.Runtime(TRT_LOGGER)

    # Creating cuda engine
    print('Creating cuda engine....')
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    print('Cuda engine created')

    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            print('input:', input_shape)

        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            print('output:', output_shape)
            # create page-locked memory buffers (i.e. won't be swapped to disk)

    return engine, context


if __name__ == '__main__':
    yolov5model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15)
    engine, context = yolov5_to_tensorrt(yolov5model)

    host_input_buffer = None
    device_input_buffer = None
    host_output_buffers = []
    device_output_buffers = []

    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:  # we expect only one input
            input_shape = engine.get_tensor_shape(binding)
            host_input_buffer = np.zeros(input_shape, dtype=np.float32)
            input_size = trt.volume(input_shape) * np.dtype(np.float32).itemsize
            device_input_buffer = cuda.mem_alloc(input_size)

        elif engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:  # and one output
            output_shape = engine.get_tensor_shape(binding)
            host_output_buffers.append(np.zeros(output_shape, dtype=np.float32))
            host_output_pagelocked = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
            device_output_buffers.append(cuda.mem_alloc(host_output_pagelocked.nbytes))

    _, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
    data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)


    evaluator = MyEvaluator()
    evaluator_complete_metric = MyEvaluatorCompleteMetric()


    stream = cuda.Stream()
    for images, targets, converted_boxes in tqdm(data_loader_test, total=len(data_loader_test), desc='Inference Tensorrt'):


        cuda.memcpy_htod_async(device_input_buffer, images.numpy(), stream)

        context.execute_async_v2([int(device_input_buffer)] + [int(i) for i in device_output_buffers], stream.handle, None)

        for host_buffer, device_buffer in zip(host_output_buffers, device_output_buffers):
            cuda.memcpy_dtoh_async(host_buffer, device_buffer, stream)

        stream.synchronize()

        preds = non_max_suppression(torch.from_numpy(host_output_buffers[-1]),
                                    0.75,
                                    0.45,

                                    multi_label=True,
                                    agnostic=True,
                                    max_det=300)

        evaluator.add_predictions_yolo(targets=targets, predictions=preds, imgs_size=[images.size()[2], images.size()[3]])
        evaluator_complete_metric.add_predictions_yolo(targets=targets, predictions=preds, imgs_size=[images.size()[2], images.size()[3]])

    metrics = evaluator.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)
    complete_metrics = evaluator_complete_metric.get_metrics(iou_threshold=0.75, confidence_threshold=0.75, door_no_door_task=False, plot_curves=False)

    mAP = 0
    print('Results per bounding box:')
    for label, values in sorted(metrics['per_bbox'].items(), key=lambda v: v[0]):
        mAP += values['AP']
        print(f'\tLabel {label} -> AP = {values["AP"]}, Total positives = {values["total_positives"]}, TP = {values["TP"]}, FP = {values["FP"]}')
        #print(f'\t\tPositives = {values["TP"] / values["total_positives"] * 100:.2f}%, False positives = {values["FP"] / (values["TP"] + values["FP"]) * 100:.2f}%')
    #print(f'\tmAP = {mAP / len(metrics["per_bbox"].keys())}')
