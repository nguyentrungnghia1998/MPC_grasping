# Create an MPC network that takes in a image, text, state, and embeded time, control signal
 
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .system_identification import SystemIdentification, TimeEmbedding
from .controller import Controller, Decoder
from .ablef import RefferenceVisionLanguage
import numpy as np
from tqdm import tqdm
from inference.models.grasp_model import LanguageGraspModel
 
# Define the model
class MPC_Grasping_Detection(LanguageGraspModel):
   
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

            self.encoder = nn.Sequential(
                nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),  # CNN layer with 16 output channels
                nn.ReLU(),  # ReLU activation function
                nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # CNN layer with 32 output channels
                nn.ReLU(),  # ReLU activation function
                nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling layer
                nn.Flatten(),  # Flatten the input tensor
                nn.Linear(32 * 56 * 56, 512)  # Linear layer to convert to (N, 512)
            )

            self.decoder = Decoder()

            self.traj_control = nn.GRUCell(input_size=512, hidden_size=768)        
           
            # Define the controller model
            self.controller = Controller(reff_dim, state_dim, control_dim, time_dim, hidden_dim)
           
           
            # Define the time embedding model
            self.time_embedding = TimeEmbedding(time_dim, time_step)
   
        def forward(self ,img, query, alpha, idx):
            img = self.refference(None, img, None, query, alpha, idx)
            z = img
            output_wp = list()
            traj_hidden_state = list()
            x = self.decoder(z)
            for i in range(self.time_step):
                x_in = self.encoder(x)
                z = self.traj_control(x_in, z)
                traj_hidden_state.append(z)
                x = self.decoder(z)
                output_wp.append(x)
            
            return img, output_wp


        def initial_state(self, batch_size, device):
            # Return a tensor of size (batch_size, state_dim)
            # x, y, w, h, theta
            # x and y are the center of the object, x and y are in the range of [0, 1]
            # w and h are the width and height of the object, w and h are in the range of [0, 1]
            # theta is the angle of the object, theta is in the range of [-pi, pi]
            # use uniform distribution to generate x, y, w, h, theta
            shape = (batch_size, 4, 224, 224)
            tensor = torch.rand(shape)

        # Assign values to each channel
            tensor[:, 0, :, :] = torch.rand(batch_size, 224, 224)  # pos_image (channel 0)
            tensor[:, 1, :, :] = torch.rand(batch_size, 224, 224) * 2 - 1  # cos_image (channel 1)
            tensor[:, 2, :, :] = torch.rand(batch_size, 224, 224) * 2 - 1  # sin_image (channel 2)
            tensor[:, 3, :, :] = torch.rand(batch_size, 224, 224)  # width_image (channel 3)
            return tensor.to(device)
 
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
       
        def compute_loss(self, label, output):
           
 
            predict = output[-1]
            pos_label, cos_label, sin_label, width_label = label

            cos_loss = F.mse_loss(predict[:,1,:,:].unsqueeze(dim = 1), cos_label)
            sin_loss = F.mse_loss(predict[:,2,:,:].unsqueeze(dim = 1), sin_label)
            width_loss = F.mse_loss(predict[:,3,:,:].unsqueeze(dim = 1), width_label)

            pos_loss = F.mse_loss(predict[:,0,:,:].unsqueeze(dim = 1), pos_label)

            return {
            'loss': pos_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': pos_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss,
                # 'contr_loss': contr_loss,
            },
            'pred': {
                'pos': predict[:,0,:,:].unsqueeze(dim = 1),
                'cos': predict[:,1,:,:].unsqueeze(dim = 1),
                'sin': predict[:,2,:,:].unsqueeze(dim = 1),
                'width': predict[:,3,:,:].unsqueeze(dim = 1)
            }
        }