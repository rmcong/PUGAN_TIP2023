import os
import cv2
import time
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image


def Weights_Normal(m):
    # initialize weights as Normal(mean, std)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(1, 3, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(3, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.cov1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()

        )
        self.cov2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.cov3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU()
        )
        self.Norm = nn.InstanceNorm2d(1)
        self.Tan = nn.Tanh()
    def forward(self, F1, F2, A):
        A = self.Norm(self.attention(A))
        dif = self.Norm(self.cov1(F2 - F1))
        A = torch.le(dif, A) * A + torch.le(A, dif) * dif
        A = self.Tan(A)
        A = F1 * A
        A = self.cov2(A)
        A = self.Tan(A)
        return A

class ConLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)#       
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConLayer(in_channels, in_channels, kernel_size=3, stride=1)
        self.conv2 = ConLayer(in_channels, in_channels, kernel_size=3, stride=1)
        self.conv3 = ConLayer(in_channels, out_channels, kernel_size=3, stride=1)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out) * 0.1)
        out = torch.add(out, residual)
        out = self.conv3(out)
        return out

class UNetDown(nn.Module):
    """ Standard UNet down-sampling block 
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self.dense = nn.Sequential(
            ResidualBlock(out_size, out_size),
            ResidualBlock(out_size, out_size),
            ResidualBlock(out_size, out_size)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_size, out_size, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        self.dense = nn.Sequential(
            ResidualBlock(out_size * 2, out_size * 2),
            ResidualBlock(out_size * 2, out_size),
            ResidualBlock(out_size, out_size)
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x1 = torch.cat((x, skip_input),1)
        x = self.dense(x1)
        return x


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)


class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """
    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake):
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True,)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty



