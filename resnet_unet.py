import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary

from ops import UNetBlock, ResNetBlock

class ResNetUnet(nn.Module):

    def __init__(self, in_channels=1, out_channels=4, init_features=64):
        super(ResNetUnet, self).__init__()

        features = [init_features * i for i in [1, 2, 4, 8]]
        self.resnet = models.resnet34(pretrained=False)

        self.unet_x = UNetBlock(in_channels, features[0], name="unet_x")

        self.encoder0 = ResNetBlock(in_channels, init_features, self.resnet)    # init_features
        self.pool1 = self.resnet.maxpool
        self.encoder1 = self.resnet.layer1          #init_features*1
        self.encoder2 = self.resnet.layer2          #init_features*2
        self.encoder3 = self.resnet.layer3          #init_features*4
        self.encoder4 = self.resnet.layer4          #init_features*8

        self.bottleneck = UNetBlock(features[3], features[3], name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder4 = UNetBlock((features[2] + features[2]), features[3], name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder3 = UNetBlock((features[2] + features[1]), features[2], name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder2 = UNetBlock((features[1] + features[0]), features[1], name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder1 = UNetBlock((features[0] + features[0]), features[0], name="dec1")
        self.upconv0 = nn.ConvTranspose2d(features[0], features[0]//2, kernel_size=2, stride=2)
        self.decoder0 = UNetBlock((features[0]//2 + features[0]), features[0], name="dec0")
        self.conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        unet0 = self.unet_x(x)
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(self.pool1(enc0))
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc3), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc1), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc0), dim=1)
        dec1 = self.decoder1(dec1)

        dec0 = self.upconv0(dec1)
        dec0 = torch.cat((dec0, unet0), dim=1)
        dec0 = self.decoder0(dec0)

        return torch.sigmoid(self.conv(dec0))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetUnet(in_channels=1)
    model = model.to(device)

    summary(model, input_size=(1, 512, 512))

    if True:
        from torch.autograd import Variable
        img = Variable(torch.rand(2, 1, 512, 512))

        net = ResNetUnet()
        out = net(img)

        print(out.size())