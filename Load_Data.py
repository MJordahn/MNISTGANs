import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from Discriminator import DiscriminatorNet
from torch.utils.data import DataLoader
from Generator import GeneratorNet
from PIL import Image
from torchvision import transforms, datasets
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
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

def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    print(vectors)
    return vectors.view(vectors.size(0), 1, 28, 28)

#To speed up training we'll only work on a subset of the data
data = np.load('Data/mnist.npz')
num_classes = 10
x_train = data['X_train'].astype('float32')
targets_train = data['y_train'][:1000].astype('int32')

x_valid = data['X_valid'][:500].astype('float32')
targets_valid = data['y_valid'][:500].astype('int32')

x_test = data['X_test'][:500].astype('float32')
targets_test = data['y_test'][:500].astype('int32')

# print("Information on dataset")
# print("x_train", x_train.shape)
# print("targets_train", targets_train.shape)
# print("x_valid", x_valid.shape)
# print("targets_valid", targets_valid.shape)
# print("x_test", x_test.shape)
# print("targets_test", targets_test.shape)

discriminator = DiscriminatorNet()
generator = GeneratorNet()

d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0001)

loss = nn.BCELoss()

#plot a few MNIST examples
idx, dim, classes = 0, 28, 10
# create empty canvas
canvas = np.zeros((dim*classes, classes*dim))

# fill with tensors
# for i in range(classes):
#     for j in range(classes):
#         canvas[i*dim:(i+1)*dim, j*dim:(j+1)*dim] = x_train[idx].reshape((dim, dim))
#         idx += 1

# visualize matrix of tensors as gray scale image
# plt.figure(figsize=(4, 4))
# plt.axis('off')
# plt.imshow(canvas, cmap='gray')
# plt.title('MNIST handwritten digits')
# plt.show()

train_ratio = 5

# Total number of epochs to train
num_epochs = 200
samples_run_through = 1
d_errors_f = open("Errors/run2/d_errors.txt","w")
g_errors_f = open("Errors/run2/g_errors.txt","w")
x_train = DataLoader(torch.from_numpy(x_train), 16, shuffle=True)
for epoch in range(num_epochs):
    for n_batch, real_batch in enumerate(x_train):
        N = real_batch.size(0)
        # 1. Train Discriminator
        real_data = Variable(real_batch)
        for i in range(train_ratio):
            # Generate fake data and detach
            # (so gradients are not calculated for generator)
            fake_data = generator(noise(N)).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = \
                  train_discriminator(d_optimizer, real_data, fake_data)


        # 2. Train Generator
        # Generate fake data
        fake_data = generator(noise(N))
        # Train G
        g_error = train_generator(g_optimizer, fake_data)

        d_errors_f.write(str(d_error.item())+";"+str(samples_run_through)+";"+str(epoch)+";"+str(n_batch)+"\n")
        g_errors_f.write(str(g_error.item())+";"+str(samples_run_through)+";"+str(epoch)+";"+str(n_batch)+"\n")

        if (n_batch) % 250 == 0:
            plt.figure(figsize=(4, 4))
            plt.axis('off')
            plt.imshow(fake_data[0].detach().numpy().reshape(28,28), cmap='gray')
            plt.title("epoch_" + str(epoch) + "_batch_" + str(n_batch))
            plt.savefig("Results_run2/" + "epoch_" + str(epoch) + "_batch_" + str(n_batch) + ".png")
        if (epoch) % 5 == 0:
            torch.save(discriminator.state_dict(), "Models/run2/discriminator_"+str(epoch))
            torch.save(generator.state_dict(), "Models/run2/generator_"+str(epoch))
