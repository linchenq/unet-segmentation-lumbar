import torch
import torch.nn as nn
from torchsummary import summary

from ops import NestedNetBlock

class NestedUnet(nn.Module):

    def __init__(self, in_channels=1, out_channels=4, init_features=32):
        super(NestedUnet, self).__init__()

        features = [init_features * i for i in [1, 2, 4, 8, 16]]

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv3 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.upconv0 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)


        self.conv0_0 = NestedNetBlock(in_channels, features[0], features[0], "conv0_0")
        self.conv1_0 = NestedNetBlock(features[0], features[1], features[1], "conv1_0")
        self.conv2_0 = NestedNetBlock(features[1], features[2], features[2], "conv2_0")
        self.conv3_0 = NestedNetBlock(features[2], features[3], features[3], "conv3_0")
        self.conv4_0 = NestedNetBlock(features[3], features[4], features[4], "conv4_0")

        self.conv0_1 = NestedNetBlock(features[0]*2, features[0], features[0], "conv0_1")
        self.conv1_1 = NestedNetBlock(features[1]*2, features[1], features[1], "conv1_1")
        self.conv2_1 = NestedNetBlock(features[2]*2, features[2], features[2], "conv2_1")
        self.conv3_1 = NestedNetBlock(features[3]*2, features[3], features[3], "conv3_1")

        self.conv0_2 = NestedNetBlock(features[0]*3, features[0], features[0], "conv0_2")
        self.conv1_2 = NestedNetBlock(features[1]*3, features[1], features[1], "conv1_2")
        self.conv2_2 = NestedNetBlock(features[2]*3, features[2], features[2], "conv2_2")

        self.conv0_3 = NestedNetBlock(features[0]*4, features[0], features[0], "conv0_3")
        self.conv1_3 = NestedNetBlock(features[1]*4, features[1], features[1], "conv0_3")

        self.conv0_4 = NestedNetBlock(features[0]*5, features[0], features[0], "conv0_4")

        self.conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x0_1 = self.conv0_1(torch.cat((x0_0, self.upconv0(x1_0)), dim=1))
        x1_1 = self.conv1_1(torch.cat((x1_0, self.upconv1(x2_0)), dim=1))
        x2_1 = self.conv2_1(torch.cat((x2_0, self.upconv2(x3_0)), dim=1))
        x3_1 = self.conv3_1(torch.cat((x3_0, self.upconv3(x4_0)), dim=1))

        x0_2 = self.conv0_2(torch.cat((x0_1, x0_0, self.upconv0(x1_1)), dim=1))
        x1_2 = self.conv1_2(torch.cat((x1_1, x1_0, self.upconv1(x2_1)), dim=1))
        x2_2 = self.conv2_2(torch.cat((x2_1, x2_0, self.upconv2(x3_1)), dim=1))

        x0_3 = self.conv0_3(torch.cat((x0_2, x0_1, x0_0, self.upconv0(x1_2)), dim=1))
        x1_3 = self.conv1_3(torch.cat((x1_2, x1_1, x1_0, self.upconv1(x2_2)), dim=1))

        x0_4 = self.conv0_4(torch.cat((x0_3, x0_2, x0_1, x0_0, self.upconv0(x1_3)), dim=1))

        out = self.conv(x0_4)
        return out

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NestedUnet(in_channels=1, out_channels=4)
    model = model.to(device)

    summary(model, input_size=(1, 512, 512))

    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 512, 512))

        net = NestedUnet()
        out = net(img)

        print(out.size())

