import os
import hdf5storage
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import config
    
class SpineDataset(Dataset):
    def __init__(self, path):
        self.input_images = np.zeros((config.D, config.C, config.H, config.W))
        self.mask_images = np.zeros((config.D, config.O, config.H, config.W))
        self.__dataload(path)
        
    def __dataload(self, path):
        if not os.path.isdir(path):
            raise FileNotFoundError('Path is not a directory')

        for i in range(0, config.D):
            MatData=hdf5storage.loadmat(path + 'SegData_P' + str(i+1)  +'.mat',
                            options=hdf5storage.Options(matlab_compatible=True))
            # enlarge the matrix to specific channels
            for j in range(0, config.C):
                self.input_images[i, j] = MatData['I']

            outputs = ('DSegMap', 'VSegMap', 'BgSegMap', 'OtherSegMap')
            for j in range(0, config.O):
                self.mask_images[i, j] = MatData[outputs[j]]
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        image = self.input_images[idx]
        mask = self.mask_images[idx]
        
        return [image, mask]
