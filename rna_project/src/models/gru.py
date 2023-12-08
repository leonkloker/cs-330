import torch
import torch.nn as nn

import models.cnn

class GRU(nn.Module):
    def __init__(self, d_features, d_model, n_layers, conv_dims=[256, 64, 16, 1], d_out=1, dropout=0.0):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size=d_features, hidden_size=d_model, 
                          num_layers=n_layers, bidirectional=True,
                          batch_first=True, dropout=dropout)
        
        conv_dims = [2*d_model] + conv_dims
        self.cnn = models.cnn.Conv1DModel(conv_dims)
                                 
    def forward(self, x):
        x, _ = self.gru(x)
        x = self.cnn(x)
        return x
