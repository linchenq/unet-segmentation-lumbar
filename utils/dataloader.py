import os
import hdf5storage
import numpy as np

from torch.utils.data import Dataset

import utils.config as cfg

class SpineDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = np.zeros((cfg.C, cfg.H, cfg.W))
        mask = np.zeros((cfg.O, cfg.H, cfg.W))
        img_path = self.img_files[idx % len(self.img_files)].rstrip()

        MatData = hdf5storage.loadmat(img_path, options=hdf5storage.Options(matlab_compatible=True))

        # enlarge the matrix to specific channels
        if MatData['I'].ndim == 2:
            for i in range(cfg.C):
                image[i, ...] = MatData['I']
        else:
            raise NotImplementedError

        # generate output channels
        for i in range(cfg.O):
            mask[i, ...] = MatData[cfg.OUTPUT_CHANNELS[i]]

        return [image, mask]

# if __name__ == "__main__":
#     path = "C:/Research/LumbarSpine/Github/unet-segmentation-lumbar/dataset/valid.txt"
#     with open(path, 'r') as file:
#         imgs = file.readlines()
#     print(len(imgs))
#     for i in range(3):
#         img_path = imgs[i % len(imgs)].rstrip()
#         MatData = hdf5storage.loadmat(img_path, options=hdf5storage.Options(matlab_compatible=True))
#         print(MatData['I'].shape)