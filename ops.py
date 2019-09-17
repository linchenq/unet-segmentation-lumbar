from collections import OrderedDict

import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, name, norm=True):
        super(UNetBlock, self).__init__()
        self.norm = norm
        
        self.block = nn.Sequential(OrderedDict([
                        (name + "conv1", nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)),
                        (name + "norm1", nn.BatchNorm2d(out_ch)),
                        (name + "relu1", nn.ReLU(inplace=True)),
                        (name + "conv2", nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)),
                        (name + "norm2", nn.BatchNorm2d(out_ch)),
                        (name + "relu2", nn.ReLU(inplace=True))
        ]))
        
        self.block_wo_norm = nn.Sequential(OrderedDict([
                        (name + "conv1_wo", nn.Conv2d(in_ch, out_ch, 3, padding=1)),
                        (name + "relu1_wo", nn.ReLU(inplace=True)),
                        (name + "conv2_wo", nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                        (name + "relu2_wo", nn.ReLU(inplace=True))
        ]))
        
    def forward(self, input):
        if self.norm:
            return self.block(input)
        else:
            return self.block_wo_norm(input)

