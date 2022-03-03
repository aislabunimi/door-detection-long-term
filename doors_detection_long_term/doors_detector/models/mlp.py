from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim for _ in range(num_layers - 1)]

        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip([input_dim] + hidden_dims, hidden_dims + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x