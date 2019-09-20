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

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, model):
        super(ResNetBlock, self).__init__()
        self.model = model

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(
                self.conv1,
                self.model.bn1,
                self.model.relu,
        )

    def forward(self, x):
        return self.encoder(x)


class AttentionNetBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionNetBlock, self).__init__()

        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_l = nn.Conv2d(F_l, F_int, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
        )

    def forward(self, g, x):
        g_new = self.W_g(g)
        x_new = self.W_l(x)

        pis_res = self.relu(g_new + x_new)
        pis_res = self.psi(pis_res)

        res = x * pis_res
        return res

if __name__ == '__main__':
    from torch.autograd import Variable

    img = Variable(torch.rand(2, 16, 32, 32))
    gat = Variable(torch.rand(2, 16, 32, 32))

    net = AttentionNetBlock(F_g=16, F_l=16, F_int=8)
    out = net(gat, img)

    print(out.size())