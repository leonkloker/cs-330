import numpy as np
import os
import sys
from torch.utils.data import Dataset
import torch

sys.path.insert(0, '../')

from models.utils import *

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

class DatasetEmbeddings(Dataset):
    def __init__(self, root_dir, mode='train', N=0):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.mode = mode
        
        if mode == 'train':
            if N == 0:
                self.length = int(len(self.file_list) * TRAIN_SPLIT)
            else:
                self.length = int(N * TRAIN_SPLIT)
        
        elif mode == 'val':
            if N == 0:
                self.length = int(len(self.file_list) * VAL_SPLIT)
            else:
                self.length = int(N * VAL_SPLIT)

        elif mode == 'test':
            if N == 0:
                self.length = int(len(self.file_list) * TEST_SPLIT)
            else:
                self.length = int(N * TEST_SPLIT)

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if self.mode == 'val':
            idx = idx + int((self.length / VAL_SPLIT) * TRAIN_SPLIT)
        if self.mode == 'test':
            idx = idx + int((self.length / TEST_SPLIT) * (TRAIN_SPLIT + VAL_SPLIT))

        npy_file = np.load(os.path.join(self.root_dir, self.file_list[idx]))
        x = npy_file["x"]
        y1 = npy_file["y1"][:x.shape[0]]
        y2 = npy_file["y2"][:x.shape[0]]
        return x, y1, y2
