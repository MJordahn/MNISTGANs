import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

class DiscriminatorNet(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 1
        n_out = 1

        self.out = nn.Sequential(
            torch.nn.Linear(n_features, n_out),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], 1)
        x = self.out(x)
        return x
