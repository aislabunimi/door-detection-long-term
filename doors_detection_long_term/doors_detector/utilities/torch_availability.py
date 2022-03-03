import torch

x = torch.rand(5, 3)
print(x)

print('Is cuda available? ' + str(torch.cuda.is_available()))