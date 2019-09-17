import argparse

from torch.utils.data import random_split
from torch.utils.data import Dataset
import torch

from dataloader import SpineDataset
from unet import UNet

DEBUG_MODE = True
RANDOM_SEED = 17

DATA_PATH='C:/Research/LumbarSpine/RealSegmentationData/'

H, W, D, C, O = 512, 512, 30, 1, 4
BATCH_SIZE = 3
NUM_EPOCHS = 5
LEARNING_RATE = 0.01
LR_SCHEDULER_STEP = 50
LOG_STEP=1

def argparser():
    parser = argparse.ArgumentParser(description="U-Net for semantic segmentation")

    # Model
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr_scheduler_step", type=int, default=LR_SCHEDULER_STEP)
    parser.add_argument("--device", type=str, default="cuda")

    # log
    parser.add_argument("--log_folder", type=str, default="./SpineUNet_Logs")
    parser.add_argument("--save_folder", type=str, default="./SpineUNet_Saves")
    parser.add_argument("--log_step", type=int, default=LOG_STEP)

    config = parser.parse_args(args=[])

    if DEBUG_MODE:
        torch.manual_seed(RANDOM_SEED)

    dataset_path =  DATA_PATH
    trainset, validset, testset = random_split(SpineDataset(DATA_PATH), lengths=[20, 5, 5])

    dataset = {
        'train': trainset,
        'valid': validset,
        'test': testset
    }

    config.h, config.w, config.d, config.c, config.num_classes = H, W, D, C, O

    model = UNet(in_channels=C, out_channels=O, init_features=32)

    return config, model, dataset



























    return args