import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import dnn

WEIGHTS_PATH='./my_fc_weights.pt'

# Create the object to preprocess the dataset
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)), 
                              ])  

# Download the dataset
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
valset = datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)

# Create the iterating objects
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# Create the DNN model
model = dnn.My_DNN()
print(model)

# Load the weights previously generated
model.load_state_dict(torch.load(WEIGHTS_PATH))
print("The weights have been loaded")

#Infiriendo una primera imagen de ejemplo:
print("Predicting a single image:")
images, labels = next(iter(valloader))
img = images[0]
img = img.view(1, 1, 28, 28)
with torch.no_grad():
    logps = model(img)
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
print("Real Digit = ", labels[0])


