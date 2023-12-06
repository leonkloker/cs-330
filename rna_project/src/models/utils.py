import math
import numpy as np
import random
import re
import torch

from torch import nn
from torch.utils.data import DataLoader, Subset

def generate_mask(sz1, sz2=None, window=-1):
        # square mask
        if sz2 is None:
            sz2 = sz1
        
        # no mask
        if window == -2:
            mask = None

        # mask when all past history is available
        elif window == -1:
            mask = (torch.tril(torch.ones(sz1, sz2)) == 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz1, sz2)
            for i in range(sz1):
                mask[i, max(0, i - window + 1) : min(i + 1, sz2)] = 1
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

def generate_mask_bool(sz1, sz2=None, window=-1):
        # square mask
        if sz2 is None:
            sz2 = sz1
        
        # no mask
        if window == -2:
            mask = None

        # mask when all past history is available
        elif window == -1:
            mask = torch.logical_not((torch.tril(torch.ones(sz1, sz2)) == 1).bool())
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz1, sz2)
            for i in range(sz1):
                mask[i, max(0, i - window + 1) : min(i + 1, sz2)] = 1
            mask = torch.logical_not(mask.bool())

        return mask

class PositionalEncodingNLP(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncodingNLP, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLinear(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super(PositionalEncodingLinear, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:,:] = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / max_len
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingSinusoidal(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super(PositionalEncodingSinusoidal, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        pos_encoding[:,:] = -torch.cos(torch.pi * (position / max_len)).unsqueeze(1)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLearned(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super(PositionalEncodingLearned, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.linear_pos_encoding = PositionalEncodingLinear(d_model, max_len, dropout)
        self.linear = nn.Sequential(nn.Linear(d_model, d_model), 
                                    nn.ReLU(), 
                                    nn.Linear(d_model, d_model),
                                    nn.Tanh())

    def forward(self, x):
        x = x + self.linear(self.linear_pos_encoding.pos_encoding)[:x.size(-2), :]
        return self.dropout(x)

class Time2Vec(nn.Module):
    def __init__(self, k, dropout=0.0):
        super(Time2Vec, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(1, k)

    def forward(self, x, t=[]):
        if len(t) == 0:
            t = torch.arange(x.size(-2), dtype=torch.float32).unsqueeze(-1)
        else:
            t = torch.tensor(t, dtype=torch.float32).unsqueeze(-1)

        t = t.to(x.device)
        t = self.linear(t)
        t = torch.cat([t[:, 0].unsqueeze(-1), torch.sin(t[:, 1:])], dim=-1)
        t = t.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([x, t], dim=-1)
        return x
    
class RelativeL2Loss(nn.Module):
    def __init__(self):
        super(RelativeL2Loss, self).__init__()

    def forward(self, prediction, ground_truth, avg_dim=[0]):
        dims = set([i for i in range(len(prediction.shape))])
        for dim in avg_dim:
            dims.remove(dim)
        dims = tuple(dims)
        
        l2_error = torch.linalg.vector_norm(prediction - ground_truth, ord=2, dim=dims)
        l2_norm = torch.linalg.vector_norm(ground_truth, ord=2, dim=dims)
        loss = torch.mean(l2_error / l2_norm)
        return loss
    
class MSEloss(nn.Module):
    def __init__(self):
        super(MSEloss, self).__init__()

    def forward(self, prediction, ground_truth):
        loss = torch.mean((prediction - ground_truth)**2)
        return loss

class CustomMAEloss(nn.Module):
    def __init__(self):
        super(CustomMAEloss, self).__init__()

    def forward(self, prediction, ground_truth):
        loss = torch.nanmean(torch.abs(prediction - ground_truth))
        return loss
    
def pearsonCorrelation(prediction, ground_truth):
    
    if ground_truth.dim() == 1:
        nan_mask = ~torch.isnan(ground_truth)
        pearson_avg = torch.corrcoef(torch.stack([prediction[nan_mask], ground_truth[nan_mask]]))[0, 1]
        pearson_med = pearson_avg
    else:
        pearson_coeffs = torch.zeros(prediction.shape[0])
        for i in range(prediction.shape[0]):
            nan_mask = ~torch.isnan(ground_truth[i])
            pearson_coeffs[i] = torch.corrcoef(torch.stack([prediction[i][nan_mask], ground_truth[i][nan_mask]]))[0, 1]
        pearson_avg = torch.mean(pearson_coeffs)
        pearson_med = torch.median(pearson_coeffs)

    return pearson_avg, pearson_med

def get_random_subset_loader(dataset, batch_size, subset_fraction=0.1):
    n = len(dataset)
    subset_indices = random.sample(range(n), int(n * subset_fraction))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader

def create_subdomain_sample_from_centers(centers):
    new_sample = np.zeros((64, centers.shape[1] * 5), dtype=np.float32)
    for i in range(64):
        center = centers[i, :]
        
        if i >= 8:
            top = centers[i-8, :]
        else:
            top = np.zeros_like(center)
            
        if i < 56:
            bottom = centers[i+8, :]
        else:
            bottom = np.zeros_like(center)
            
        if i % 8 != 0:
            left = centers[i-1, :]
        else:
            left = np.zeros_like(center)
            
        if (i+1) % 8 != 0:
            right = centers[i+1, :]
        else:
            right = np.zeros_like(center)
            
        new_sample[i, :] = np.concatenate((center, top, right, bottom, left))
        
    return new_sample

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def get_window_size(nenc, ndec, enc_window, dec_window, mem_window):
    return max(nenc * (enc_window - 1) + mem_window - 1, dec_window - 1,
                    nenc * (enc_window - 1) + mem_window - 1 + (dec_window - 1) * (ndec - 1))
