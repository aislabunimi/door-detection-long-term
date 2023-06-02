import onnx
import tensorrt
import torch
import tensorrt as trt

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.model_names import YOLOv5
from doors_detection_long_term.doors_detector.models.yolov5 import YOLOv5Model, \
    EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15


def yolov5_to_tensorrt(model: YOLOv5Model, input_tensor: torch.Tensor=torch.ones(1, 3, 320, 320), fp16: bool=False):

    # Export to ONNX format
    model_path = 'yolov5/model_onnx.onnx'

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








if __name__ == '__main__':
    yolov5model = YOLOv5Model(model_name=YOLOv5, n_labels=2, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15)
    yolov5_to_tensorrt(yolov5model)
