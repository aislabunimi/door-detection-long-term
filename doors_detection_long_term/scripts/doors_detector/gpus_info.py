import torch

# Print the gpu list

for i in range(torch.cuda.device_count()):
    print(f'Gpu {i} -> {torch.cuda.get_device_name(i)} - {round(torch.cuda.get_device_properties(i).total_memory / 1024**(3), 2)}GB')