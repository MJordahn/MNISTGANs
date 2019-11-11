import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 1
        n_out = 1

        self.out = nn.Sequential(
            nn.Linear(n_features, n_out, bias=False),
        )

    def forward(self, x):
        x = self.out(x)
        return x
