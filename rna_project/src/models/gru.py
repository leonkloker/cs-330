import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, d_features, d_model, n_layers, d_out=1, dropout=0.0):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size=d_features, hidden_size=d_model, 
                          num_layers=n_layers, bidirectional=True,
                          batch_first=True, dropout=dropout)

        self.fc = nn.Sequential(nn.Linear(d_model, 64),
                                      nn.Dropout(dropout),
                                      nn.GELU(),
                                      nn.Linear(64, 16),
                                      nn.Dropout(dropout),
                                      nn.GELU(),
                                      nn.Linear(16, 1),
                                      nn.Sigmoid())

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.fc(x)
        return x
