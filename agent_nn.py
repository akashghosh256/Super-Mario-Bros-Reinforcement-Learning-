import torch
from torch import nn
import numpy as np

class AgentNN(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),  # Convolutional layer with 32 filters, kernel size of 8x8, and stride of 4
            nn.ReLU(),  # ReLU activation function
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # Convolutional layer with 64 filters, kernel size of 4x4, and stride of 2
            nn.ReLU(),  # ReLU activation function
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # Convolutional layer with 64 filters, kernel size of 3x3, and stride of 1
            nn.ReLU(),  # ReLU activation function
        )

        # Calculate the output size of the convolutional layers
        conv_out_size = self._get_conv_out(input_shape)

        # Linear layers
        self.network = nn.Sequential(
            self.conv_layers,  # Add convolutional layers
            nn.Flatten(),  # Flatten the output from convolutional layers
            nn.Linear(conv_out_size, 512),  # Linear layer with input size conv_out_size and output size 512
            nn.ReLU(),  # ReLU activation function
            nn.Linear(512, n_actions)  # Output layer with input size 512 and output size n_actions
        )

        # If freeze is True, freeze the parameters of the network
        if freeze:
            self._freeze()
        
        # Move the network to GPU if available, else use CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        return self.network(x)

    def _get_conv_out(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        # Calculate the output size of the convolutional layers
        return int(np.prod(o.size()))
    
    def _freeze(self):        
        # Freeze the parameters of the network
        for p in self.network.parameters():
            p.requires_grad = False
