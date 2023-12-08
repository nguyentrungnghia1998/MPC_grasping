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
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, control_dim)
 
    def forward(self, x, r, t):
        y = torch.cat((x, r, t), 1)
        y = F.relu(self.bn1(self.fc1(y)))
        y = F.relu(self.bn2(self.fc2(y)))
        y = self.fc3(y)
        return y
    
class Decoder(nn.Module):
    # Decoder a Tensor [batch_size, 768] to a Tensor [batch_size, 4, 224, 224]
    # Channel 0: 0 to 1
    # Channel 1: -1 to 1
    # Channel 2: -1 to 1
    # Channel 3: 0 to 1

    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(768, 64*14*14)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 2, stride=2),
        )
        self.pos = nn.Sequential(
            nn.Conv2d(4, 1, 3, padding = 1),
            nn.Sigmoid()
        )

        self.cos = nn.Sequential(
            nn.Conv2d(4, 1, 3, padding = 1),
            nn.Tanh()
        )

        self.sin = nn.Sequential(
            nn.Conv2d(4, 1, 3, padding = 1),
            nn.Tanh()
        )

        self.width = nn.Sequential(
            nn.Conv2d(4, 1, 3, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 64, 14, 14)
        x = self.decoder(x)
        pos_img = self.pos(x)
        cos_img = self.cos(x)
        sin_img = self.sin(x)
        width_img = self.width(x)
        x = torch.cat([pos_img, cos_img, sin_img, width_img], dim = 1)
        return x