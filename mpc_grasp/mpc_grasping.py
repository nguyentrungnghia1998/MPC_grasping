# Create an MPC network that takes in a image, text, state, and embeded time, control signal
 
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .system_identification import SystemIdentification, TimeEmbedding
from .controller import Controller
from .ablef import RefferenceVisionLanguage
import numpy as np
from tqdm import tqdm
 
# Define the model
class MPC_Grasping_Detection(nn.Module):
   
        def __init__(self, input_channels, reff_dim, state_dim, control_dim, time_dim, hidden_dim, dropout, prob, channel_size, time_step):
            super(MPC_Grasping_Detection, self).__init__()
            self.state_dim = state_dim
            self.control_dim = control_dim
            self.time_dim = time_dim
            self.hidden_dim = hidden_dim
            self.time_step = time_step
 
            # Define the refference signal model
            self.refference = RefferenceVisionLanguage(input_channels=input_channels,  
                                                       dropout=dropout,
                                                       prob=prob,
                                                       channel_size=channel_size)
           
            # Define the system identification model
            self.system_identification = SystemIdentification(reff_dim, state_dim, control_dim, time_dim, hidden_dim)
           
            # Define the controller model
            self.controller = Controller(reff_dim, state_dim, control_dim, time_dim, hidden_dim)
           
           
            # Define the time embedding model
            self.time_embedding = TimeEmbedding(time_dim, time_step)
   
        def forward(self, x_old, t ,img, query, alpha, idx):
            img = self.refference(None, img, None, query, alpha, idx)
            time = self.time_embedding(t)
            u = self.controller(x_old, img, time)
            x_old = self.system_identification(x_old, u, img, time)
            return x_old, u
       
 
        def initial_state(self, batch_size, device):
            # Return a tensor of size (batch_size, state_dim)
            # x, y, w, h, theta
            # x and y are the center of the object, x and y are in the range of [0, 1]
            # w and h are the width and height of the object, w and h are in the range of [0, 1]
            # theta is the angle of the object, theta is in the range of [-pi, pi]
            # use uniform distribution to generate x, y, w, h, theta
            x = torch.randn(batch_size, 1, device=device)
            y = torch.randn(batch_size, 1, device=device)
            w = torch.randn(batch_size, 1, device=device)
            h = torch.randn(batch_size, 1, device=device)
            theta = torch.randn(batch_size, 1, device=device)
            return torch.cat((x, y, w, h, theta), 1)
 
        # Get output of the system identification model to input for next time step
        # Initial state is a tensor of size (batch_size, state_dim)
        # Get output is 2 sequence of tenser (state_seq, control_seq)
        # state_seq is a tensor of size (batch_size, time_step + 1, state_dim)
        # control_seq is a tensor of size (batch_size, time_step, control_dim)
 
        def get_output(self, initial_state, img, query, alpha, idx):
            # initial_state is a tensor of size (batch_size, state_dim)
            # time_step is an integer
            # img is a tensor of size (batch_size, 3, 224, 224)
            # query is a list of string
            # alpha is a float
            # idx is an integer
            # state_seq is a tensor of size (batch_size, time_step + 1, state_dim)
            # control_seq is a tensor of size (batch_size, time_step, control_dim)
            # Save state and control sequence in numpy array, cpu
            batch_size = initial_state.shape[0]
            device = initial_state.device
            # state_seq = np.zeros((batch_size, self.time_step + 1, self.state_dim))
            # control_seq = np.zeros((batch_size, self.time_step, self.control_dim))
            # Convert initial_state to cpu
            x_old = initial_state
            # state_seq[:, 0, :] = initial_state.cpu().numpy()
            # Convert img to cpu
            for i in range(self.time_step):
                t = torch.ones(batch_size, 1, device=device)*i
                x_new, u = self.forward(x_old, t, img, query, alpha, idx)
                x_old = x_new.detach()
 
            return x_old
       
        def compute_loss(self, x_label, x_pred, u_pred):
           
 
 
            # if sample is None:
            #     pos_pred, cos_pred, sin_pred, width_pred = self.pos_output_str, self.cos_output_str, self.sin_output_str, self.width_output_str
            # else:
            #     pos_pred = sample
            #     cos_pred, sin_pred, width_pred = self.cos_output_str, self.sin_output_str, self.width_output_str
            x_loss = F.mse_loss(torch.sigmoid(x_label), x_pred)
            u_loss = F.mse_loss(u_pred, torch.zeros_like(u_pred, device=u_pred.device))
 
            # Get contrastive loss
            # contr_loss = self._get_contrastive_loss(self.full_image_atts.to(y_pos.device), y_pos)
 
            return {
                'loss': x_loss,
                'losses': {
                    'x_loss': x_loss,
                    'u_loss': u_loss
                },
                'pred': {
                    'x_pred': x_pred,
                }
            }
