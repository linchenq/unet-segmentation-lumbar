import argparse
import os

import torch

from utils.dataloader import SpineDataset
from nets.unet import UNet
from nets.resnet_unet import ResNetUnet
from nets.attention_unet import AttentionUnet
from nets.nested_unet import NestedUnet

# DATA_PATH
DATA_PATH='C:/Research/LumbarSpine/OriginalData/RealSegmentationData/'

# Image H/W/C
H, W, C = 512, 512, 1

# Network Ouput
O = 4

OUTPUT_CHANNELS = ['DSegMap', 'VSegMap', 'BgSegMap', 'OtherSegMap']

# Data Objects
D = 1890

# TRAINING SET
BATCH_SIZE = 2

NUM_EPOCHS = 150

LEARNING_RATE = 0.01

LR_SCHEDULER_STEP = 40

RANDOM_SEED = 100

# SAVE AND LOG
SAVE_STEP = 5

EVAL_STEP = 1

# DEBUG MODE
DEBUG_MODE = True

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
    parser.add_argument("--eval_step", type=int, default=EVAL_STEP)
    parser.add_argument("--model_name", type=str, default="unet")

    config = parser.parse_args(args=[])

    if DEBUG_MODE:
        torch.manual_seed(RANDOM_SEED)

    dataset_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'dataset')
    trainset, validset, testset = None, None, None
    dataset = {
        'train': trainset,
        'valid': validset,
        'test': testset
    }
    for name in ['train', 'valid', 'test']:
        set_path = os.path.join(dataset_path, f"{name}.txt")
        dataset[name] = SpineDataset(list_path=set_path)

    config.h, config.w, config.d, config.c, config.num_classes = H, W, D, C, O

    config.save_step = SAVE_STEP

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