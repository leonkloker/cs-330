import torch
import torch.nn as nn

class Conv1DModel(nn.Module):
    def __init__(self, dimensions, kernel_size=3, padding=1, dropout=0.0, activation='gelu'):
        super(Conv1DModel, self).__init__()

        # Define the convolutional layers
        self.conv_layers = nn.ModuleList()
        self.activation = nn.GELU()
        self.conv_layers.append(nn.Conv1d(in_channels=dimensions[0], out_channels=dimensions[1], kernel_size=kernel_size, padding=padding))
        for i in range(1, len(dimensions) - 1):
            self.conv_layers.append(self.activation)
            self.conv_layers.append(nn.Conv1d(in_channels=dimensions[i], out_channels=dimensions[i+1], kernel_size=kernel_size, padding=padding))
        self.conv_layers.append(nn.Sigmoid())
    
    def forward(self, x):
        # PyTorch's Conv1D expects input of shape (batch_size, channels, sequence_length)
        # Transpose the input to match this shape
        x = x.transpose(1, 2)

        # Apply the convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Transpose back to (batch_size, sequence_length, channels)
        x = x.transpose(1, 2)

        return x
