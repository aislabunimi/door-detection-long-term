import pycuda.autoinit
import numpy as np
import onnx
import tensorrt
import torch
import tensorrt as trt

def model_to_tensorrt(model, input_tensor: torch.Tensor=torch.ones(1, 3, 320, 320), fp16: bool=False):

    model.eval()

    # Export to ONNX format
    model_path = 'model_onnx.onnx'

    torch.onnx.export(model, input_tensor, model_path, input_names=['input'],
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

    return engine, context