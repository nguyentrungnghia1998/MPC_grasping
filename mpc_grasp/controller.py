# I want to create a model with controller 3 inputs and 1 outputs
# Reference signal: reff_dim
# State signal: state_dim
# Control signal: control_dim
# Time signal: time_dim
 
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
# Define the model
class Controller(nn.Module):
 
    def __init__(self, reff_dim, state_dim, control_dim, time_dim, hidden_dim):
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(reff_dim + state_dim + time_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, control_dim)
 
    def forward(self, x, r, t):
        y = torch.cat((x, r, t), 1)
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y
