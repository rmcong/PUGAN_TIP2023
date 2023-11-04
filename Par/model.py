import time
import torch
import random
import imageio
import numpy as np
from math import exp, log
from PIL import Image
import torch.nn as nn
import skimage.transform
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader


def expand(ae, depth):
    x1 = depth
    x2 = depth
    x3 = depth
    a = ae.size()
    i = a[0]
    for i in range(i):
        x1[i][0].mul(ae[i][0])
        x2[i][0].mul(ae[i][1])
        x3[i][0].mul(ae[i][2])
    x = torch.cat([x1, x2, x3], 1)
    return x

class RES(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, pad):
        super(RES, self).__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, bias=False)
        self.re = nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(out_ch)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.re(y)
        return torch.cat([y, x], 1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        layers1 = [
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        layers2 = [
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        ]
        layers3 = [
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
        ]
        layers4 = [
            nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(3),
            nn.ReLU(inplace=True),
        ]
        self.model1 = nn.Sequential(*layers1)
        self.model2 = nn.Sequential(*layers2)
        self.model3 = nn.Sequential(*layers3)
        self.model4 = nn.Sequential(*layers4)
    def forward(self, x):
        x = self.model4(self.model3(self.model2(self.model1(x))))
        return x

class preAE(torch.nn.Module):
    def __init__(self):
        super(preAE,self).__init__()
        self.conv_layer1=torch.nn.Conv2d(1,32,kernel_size=3,padding=1,bias=False)
        self.conv_layer2=torch.nn.Conv2d(32,64,kernel_size=3,padding=1,bias=False)
        self.conv_layer3=torch.nn.Conv2d(64,32,kernel_size=3,padding=1,bias=False)
        self.lin1=torch.nn.Linear(32768,128)
        self.pooling=torch.nn.MaxPool2d(2)
        self.lin2=torch.nn.Linear(128,1)
        self.re=nn.ReLU()
    def forward(self,x):
        batch_size=x.size(0)
        x=self.re(self.pooling(self.conv_layer1(x)))
        x=self.re(self.pooling(self.conv_layer2(x)))
        x=self.re(self.pooling(self.conv_layer3(x)))
        x=x.view(batch_size,-1)
        x=self.lin2(self.re(self.lin1(x)))
        return x
class AENet(torch.nn.Module):
    def __init__(self):
        super(AENet,self).__init__()
        self.preR=preAE()
        self.preG=preAE()
        self.preB=preAE()
    def forward(self,x):
        batch_size=x.size(0)
        out=[]
        X=torch.chunk(x,batch_size,dim=0)
        for i in range (batch_size):
            y=torch.chunk(X[i],3,dim=1)
            xr=self.preR(y[0])
            xg=self.preG(y[1])
            xb=self.preB(y[2])
            out.append(torch.cat([xr,xg,xb],1))
        return torch.cat(out,0)

class DNet(nn.Module):
    def __init__(self):
        super(DNet, self).__init__()
        self.net = Net()
        layers1 = [
            nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(3),
            nn.ReLU(inplace=True),
        ]
        layers2 = [
            nn.Conv2d(3, 1, kernel_size=3, padding=1, bias=False),
            #nn.InstanceNorm2d(3),
            nn.Sigmoid(),
        ]
        self.model1 = nn.Sequential(*layers1)
        self.model2 = nn.Sequential(*layers2)
    def forward(self,x):
        x1 = self.net(x)
        depth = self.model2(self.model1(x1))
        return depth

class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.res_1 = RES(4, 4, 3, 1)
        self.conv1x3 = torch.nn.Conv2d(8, 1, kernel_size=1)
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, ae, depth):
        ae1 = expand(ae, depth)
        x = torch.cat([depth,ae1],1)
        x1 = self.res_1(x)
        x4 = self.conv1x3(x1)
        return x4

class TtoDNet(nn.Module):
    def __init__(self):
        super(TtoDNet, self).__init__()
        self.res_1 = RES(1, 3, 3, 1)
        self.res_2 = RES(4, 16, 3, 1)
        layers1 = [
            nn.Conv2d(20, 8, kernel_size=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(inplace=True),
        ]
        self.model1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(8, 1, kernel_size=1),
        ]
        self.model2 = nn.Sequential(*layers2)
        self.tnet = TNet()

    def forward(self,ae, depth):
        a = self.tnet(ae, depth)
        x = self.res_1(a)
        x = self.res_2(x)
        x4 = self.model2(self.model1(x))
        return a, x4


