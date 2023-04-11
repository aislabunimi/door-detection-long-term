import time
from timeit import timeit

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import torch
from torch.utils.data import DataLoader

from doors_detection_long_term.doors_detector.utilities.utils import collate_fn, collate_fn_yolov5
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import \
    get_final_doors_dataset_epoch_analysis, get_final_doors_dataset_real_data

f = open("resnet_engine_pytorch.trt", "rb")
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

print(engine.get_binding_shape(0))
print(engine.get_binding_shape(1))
print(engine.get_binding_shape(2))


print(context.get_binding_shape(0))
print(context.get_binding_shape(1))

import numpy as np

# need to set input and output precisions to FP16 to fully enable it
output_1 = np.zeros([1, 3, 40, 40, 7], dtype = np.float16)
output_2 = np.zeros([1, 3, 20, 20, 7], dtype = np.float16)
output_3 = np.zeros([1, 3, 10, 10, 7], dtype = np.float16)
output_4 = np.zeros([1, 6300, 7], dtype = np.float16)

# allocate device memory
d_input = cuda.mem_alloc(1 * np.zeros((3, 320, 320)).nbytes)

d_output_1 = cuda.mem_alloc(1 * output_1.nbytes)
d_output_2 = cuda.mem_alloc(1 * output_2.nbytes)
d_output_3 = cuda.mem_alloc(1 * output_3.nbytes)
d_output_4 = cuda.mem_alloc(1 * output_4.nbytes)

bindings = [int(d_input), int(d_output_1), int(d_output_2),int(d_output_3), int(d_output_4)]
print(bindings)

stream = cuda.Stream()

def predict(batch): # result gets copied into output
    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)
    # execute model
    context.execute_async_v2(bindings, stream.handle, None)
    # transfer predictions back
    #print(d_output)
    cuda.memcpy_dtoh_async(output_1, d_output_1, stream)
    cuda.memcpy_dtoh_async(output_2, d_output_2, stream)
    cuda.memcpy_dtoh_async(output_3, d_output_3, stream)
    cuda.memcpy_dtoh_async(output_3, d_output_4, stream)
    # syncronize threads
    stream.synchronize()

    return output_1


train, test, labels, COLORS = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)

dataLoader = DataLoader(test, batch_size=1, collate_fn=collate_fn_yolov5, drop_last=False, num_workers=4)
start = time.time()
for img, _, _ in dataLoader:


    predict(img.numpy())

end = time.time()
print(f'duration: {end-start}, FPS: {len(dataLoader)/(end-start)}')