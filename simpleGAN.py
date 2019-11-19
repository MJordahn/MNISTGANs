import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from DiscriminatorSIMP import DiscriminatorNet
from torch.utils.data import DataLoader
from GeneratorSIMP import GeneratorNet
from PIL import Image
from torchvision import transforms, datasets
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.from_numpy(np.random.normal(0, 1, 1))).flat()
    return n

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N) )
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(mu, fake_data, discriminator):
    N = fake_data.size(0)
    learning_step = 0.01
    sum = 0
    predictions = discriminator(fake_data)
    zipped = zip(fake_data, predictions)
    for row in zipped:
        sum += row[1].data*(mu-row[0].data)

    sum = sum/N

    mu = mu - learning_step*(sum.data)
    return mu

#Get 1000 samples from normal distribution with mean 0, variance 1
data = np.random.normal(0, 1, 10000)



train_ratios = [1, 2, 3, 4, 5, 10, 20, 50]




num_epochs = 50
samples_run_through = 1
d_errors_f = open("Errors/simple"+str(num_epochs)+ "_lr001/d_errors.txt","w")
g_errors_f = open("Errors/simple"+ str(num_epochs)+ "_lr001/g_errors.txt","w")
x_train = DataLoader(torch.from_numpy(data).float(), 512, shuffle=True)
for train_ratio in train_ratios:
    torch.manual_seed(0)
    mu = torch.Tensor([2])
    discriminator = DiscriminatorNet()
    discriminator = discriminator.float()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
    loss = nn.BCELoss()
    print(train_ratio)
    for epoch in range(num_epochs):
        for n_batch, real_batch in enumerate(x_train):
            N = real_batch.size(0)
            # 1. Train Discriminator
            real_data = Variable(real_batch)
            for i in range(train_ratio):
                # Generate fake data and detach
                # (so gradients are not calculated for generator)
                fake_data = Variable(torch.from_numpy(np.random.normal(mu, 1, N))).float()
                # Train D
                d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
            samples_run_through += 1
            params = list(discriminator.parameters())
            d_errors_f.write(str(params[0].item())+";" + str(params[1].item()) + ";" +str(samples_run_through)+";"+str(epoch)+";"+str(n_batch)+";" + str(train_ratio) + "\n")
            g_errors_f.write(str(mu.item())+";"+str(samples_run_through)+";"+str(epoch)+";"+ str(train_ratio) +"\n")
            # 2. Train Generator
            # Generate fake data
            fake_data = Variable(torch.from_numpy(np.random.normal(mu, 1, N))).float()
            # Train G
            mu = train_generator(mu, fake_data, discriminator)
