import torch
import torch.nn as nn
from torchsummary import summary

from ops import UNetBlock

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=4, init_features=32, maxpool=True):
        super(UNet, self).__init__()

        self.maxpool = maxpool
        features = [init_features * i for i in [1, 2, 4, 8, 16]]

        self.encoder1 = UNetBlock(in_channels, features[0], name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1_conv = nn.Conv2d(features[0], features[0], kernel_size=2, stride=2)
        self.encoder2 = UNetBlock(features[0], features[1], name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_conv = nn.Conv2d(features[1], features[1], kernel_size=2, stride=2)
        self.encoder3 = UNetBlock(features[1], features[2], name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3_conv = nn.Conv2d(features[2], features[2], kernel_size=2, stride=2)
        self.encoder4 = UNetBlock(features[2], features[3], name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4_conv = nn.Conv2d(features[3], features[3], kernel_size=2, stride=2)

        self.bottleneck = UNetBlock(features[3], features[4], name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder4 = UNetBlock((features[3]+features[3]), features[3], name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = UNetBlock((features[2]+features[2]), features[2], name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = UNetBlock((features[1]+features[1]), features[1], name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = UNetBlock(features[1], features[0], name="dec1")

        self.conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        if self.maxpool:
            enc2 = self.encoder2(self.pool1(enc1))
            enc3 = self.encoder3(self.pool2(enc2))
            enc4 = self.encoder4(self.pool3(enc3))

            bottleneck = self.bottleneck(self.pool4(enc4))

        else:
            enc2 = self.encoder2(self.pool1_conv(enc1))
            enc3 = self.encoder3(self.pool2_conv(enc2))
            enc4 = self.encoder4(self.pool3_conv(enc3))

            bottleneck = self.bottleneck(self.pool4_conv(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)

    summary(model, input_size=(1, 512, 512))

    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 512, 512))

        net = UNet(maxpool=False)
        out = net(img)

        print(out.size())