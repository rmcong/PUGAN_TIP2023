import torch
import torch.nn as nn
import torch.nn.functional as F
from .commons import UNetUp, UNetDown, Fusion
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import disk
import skimage.filters.rank as sfr
from torchvision.utils import save_image

def inv(x , t):
    fx = []
    a = []
    t = 1.0 / t
    for j in range(x.size(0)):
        for i in range(3):
            fx.append(x[j][i] * t[j][0])
        a.append(torch.cat([fx[0].unsqueeze(0), fx[1].unsqueeze(0), fx[2].unsqueeze(0)], 0).unsqueeze(0))
    J = torch.cat(a, 0)
    return J


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)

        self.down11 = UNetDown(3, 64, normalize=False)
        self.down22 = UNetDown(64, 128)
        self.down33 = UNetDown(128, 256)
        self.down44 = UNetDown(256, 512, dropout=0.5)

        self.up4 = UNetUp(512, 512, dropout=0.5)
        self.up3 = UNetUp(512, 256)
        self.up2 = UNetUp(256, 128)
        self.up1 = UNetUp(128, 64)

        self.fs4 = Fusion(512, 512)
        self.fs3 = Fusion(256, 256)
        self.fs2 = Fusion(128, 128)
        self.fs1 = Fusion(64, 64)

        self.pool = nn.MaxPool2d(3, stride = 2, padding = 1)

        self.contr5 = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 1, 3, padding=1),
            nn.Tanh()
        )

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x, j):
        J = inv(x,j)

        j1 = self.pool(j)
        j2 = self.pool(j1)
        j3 = self.pool(j2)
        j4 = self.pool(j3)
        j5 = self.pool(j4)

        #3-64
        d1 = self.down1(x)
        dj1 = self.down11(J)

        #64-128
        d2 = self.down2(d1)
        dj2 = self.down22(dj1)

        #128-256
        d3 = self.down3(d2)
        dj3 = self.down33(dj2)

        #256-512
        d4 = self.down4(d3)
        dj4 = self.down44(dj3)

        #512-512
        d5 = self.down5(d4)

        d5 = d5 * self.contr5(j5)

        # 512-512
        d4 = self.fs4(d4, dj4, j4)
        u4 = self.up4(d5, d4)

        #512-256
        d3 = self.fs3(d3, dj3, j3)
        u3 = self.up3(u4, d3)

        #256-128
        d2 = self.fs2(d2, dj2, j2)
        u2 = self.up2(u3, d2)

        #128-64
        d1 = self.fs1(d1, dj1, j1)
        u1 = self.up1(u2, d1)

        return self.final(u1)


