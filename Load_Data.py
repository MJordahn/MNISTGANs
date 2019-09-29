import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from Discriminator import DiscriminatorNet
from Generator import GeneratorNet

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

#To speed up training we'll only work on a subset of the data
data = np.load('Data/mnist.npz')
num_classes = 10
x_train = data['X_train'].astype('float32')
targets_train = data['y_train'].astype('int32')

x_valid = data['X_valid'].astype('float32')
targets_valid = data['y_valid'].astype('int32')

x_test = data['X_test'].astype('float32')
targets_test = data['y_test'].astype('int32')

print("Information on dataset")
print("x_train", x_train.shape)
print("targets_train", targets_train.shape)
print("x_valid", x_valid.shape)
print("targets_valid", targets_valid.shape)
print("x_test", x_test.shape)
print("targets_test", targets_test.shape)

discriminator = DiscriminatorNet()
generator = GeneratorNet()
