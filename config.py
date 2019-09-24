import argparse

from torch.utils.data import random_split
import torch

from dataloader import SpineDataset
from unet import UNet
from resnet_unet import ResNetUnet
from attention_unet import AttentionUnet
from nested_unet import NestedUnet

DEBUG_MODE = True
RANDOM_SEED = 100

DATA_PATH='C:/Research/LumbarSpine/RealSegmentationData/'

H, W, D, C, O = 512, 512, 30, 1, 4
BATCH_SIZE = 3
NUM_EPOCHS = 100
LEARNING_RATE = 0.01
LR_SCHEDULER_STEP = 40
LOG_STEP = 1

def argparser():
    parser = argparse.ArgumentParser(description="U-Net for semantic segmentation")

    # Model
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr_scheduler_step", type=int, default=LR_SCHEDULER_STEP)
    parser.add_argument("--device", type=str, default="cuda")

    # log
    parser.add_argument("--log_folder", type=str, default="./logs")
    parser.add_argument("--save_folder", type=str, default="./saves")
    parser.add_argument("--log_step", type=int, default=LOG_STEP)
    parser.add_argument("--model_name", type=str, default="unet")

    config = parser.parse_args(args=[])

    if DEBUG_MODE:
        torch.manual_seed(RANDOM_SEED)

    dataset_path =  DATA_PATH
    trainset, validset, testset = random_split(SpineDataset(dataset_path), lengths=[20, 5, 5])

    dataset = {
        'train': trainset,
        'valid': validset,
        'test': testset
    }

    config.h, config.w, config.d, config.c, config.num_classes = H, W, D, C, O

    if config.model_name == "unet":
        model = UNet(in_channels=C, out_channels=O, init_features=32, maxpool=False)
    # This specific performance shows that regarding this training set:
    #      maxpooling > (better than) conv with stride=2
    elif config.model_name == "resnet_unet":
        model = ResNetUnet(in_channels=C, out_channels=O, init_features=64)
    elif config.model_name == "attention_unet":
        model = AttentionUnet(in_channels=C, out_channels=O, init_features=32)
    elif config.model_name == "nested_net":
        model = NestedUnet(in_channels=C, out_channels=O, init_features=32)
    else:
        raise NotImplementedError

    return config, model, dataset