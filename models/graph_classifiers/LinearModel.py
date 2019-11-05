import torch
import torch.nn.functional as F
from torch.nn import Linear

class LinearModel(torch.nn.Module):

    def __init__(self, dim_features, dim_target, config):
        super(LinearModel, self).__init__()

        self.out = Linear(dim_features, dim_target)

    def forward(self, x, batch):
        return self.out(x)