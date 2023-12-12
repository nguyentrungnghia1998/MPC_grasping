# I want to create a Pytorch model with 4 inputs and 1 outputs
# Reference signal: reff_dim
# State signal: state_dim
# Control signal: control_dim
# Time signal: time_dim
# Output signal: state_dim
 
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
 
 
# Define the model
# Create a time embedding for the time series data, dim = 64, 0<=T<=100
# Input is a tensor of size (batch_size, 1)
# Output is a tensor of size (batch_size, 64)
# Using Sinusoidal Time Embedding
 
import torch
import torch.nn as nn
 
class TimeEmbedding(nn.Module):
    def __init__(self, dim = 64, max_period=100):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.max_period = max_period
 
    def forward(self, x):
        # x is a tensor of size (batch_size, 1)
        # y is a tensor of size (batch_size, 64)
        y = torch.zeros(x.shape[0], self.dim, device = x.device)
        for i in range(self.dim):
            if i % 2 == 0:
                y[:, i] = torch.sin(x[:, 0] / self.max_period**(2*i/self.dim))
            else:
                y[:, i] = torch.cos(x[:, 0] / self.max_period**(2*i/self.dim))
        return y
     
 
class SystemIdentification(nn.Module):
 
    def __init__(self, reff_dim, state_dim, control_dim, time_dim, hidden_dim):
        super(SystemIdentification, self).__init__()
        self.fc1 = nn.Linear(reff_dim + state_dim + control_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
 
    def forward(self, x, u, r):
        y = torch.cat((x, u, r), 1)
        y = F.relu(self.bn1(self.fc1(y)))
        y = F.relu(self.bn2(self.fc2(y)))
        y = self.fc3(y)
        return y
