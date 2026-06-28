import numpy as np
import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None: mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, input):
        return self.block(input)
    
class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp = nn.MaxPool2d(2)

    def forward(self, input):
        return self.mp(input)
    
class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.us = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, input):
        return self.us(input)

class Denoiser(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 512, mid_channels=1024)

        self.up1 = DoubleConv(1024, 256, 512)
        self.up2 = DoubleConv(512, 128, 256)
        self.up3 = DoubleConv(256, 64, 128)
        self.up4 = DoubleConv(128, 64)

        self.maxpool = MaxPool()
        self.upsample = UpSample()
        self.out = nn.Conv2d(64, 3, kernel_size=1)


    def forward(self, input):
        down1 = self.down1(input)
        down1_1 = self.maxpool(down1)
        down2 = self.down2(down1_1)
        down2_1 = self.maxpool(down2)
        down3 = self.down3(down2_1)
        down3_1 = self.maxpool(down3)
        down4 = self.down4(down3_1)
        down4_1 = self.maxpool(down4)

        bot = self.bottleneck(down4_1)
        bot_1 = self.upsample(bot)

        up1 = self.up1(torch.cat([bot_1, down4], dim=1))
        up1_1 = self.upsample(up1)
        up2 = self.up2(torch.cat([up1_1, down3], dim=1))
        up2_1 = self.upsample(up2)
        up3 = self.up3(torch.cat([up2_1, down2], dim=1))
        up3_1 = self.upsample(up3)
        up4 = self.up4(torch.cat([up3_1, down1], dim=1))

        out = self.out(up4)

        return nn.ReLU()(out)

