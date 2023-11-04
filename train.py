import os
import sys
import yaml
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from nets.fusion import PUGAN, Gradient_Difference_Loss
from nets.commons import Weights_Normal, Gradient_Penalty
from utils.data_utils import GetTrainingPairs, GetValImage
import matplotlib.pyplot as plt
from Par.model import DNet, TtoDNet, AENet
from math import exp


## get configs and training options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="train.yaml")
args = parser.parse_args()


with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_name = cfg["dataset_name"]
dataset_path = cfg["dataset_path"]
channels = cfg["chans"]
img_width = cfg["im_width"]
img_height = cfg["im_height"]
val_interval = cfg["val_interval"]
ckpt_interval = cfg["ckpt_interval"]
from_epoch = cfg["from_epoch"]
num_epochs = cfg["num_epochs"]
batch_size = cfg["batch_size"]
lr_rate = cfg["lr_rate"]
num_critic = cfg["num_critic"]
lambda_gp = cfg["gp_weight"]
lambda_gp1 = cfg["gp1_weight"]
lambda_1 = cfg["l1_weight"]
lambda_2 = cfg["lg_weight"]
lambda_d = cfg["lambda_d"]
lambda_d1 = cfg["lambda_d1"]
model_path_d = cfg["path_d"]
model_path_t = cfg["path_t"]
model_path_a = cfg["path_a"]
model_path_g_from = cfg["path_g"]
model_path_d_from = cfg["path_D"]
model_path_d1_from = cfg["path_D1"]
version = cfg["version"]

## create dir for model
checkpoint_dir = "checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

L1_G = torch.nn.L1Loss()  # l1 loss term
L1_gp = Gradient_Penalty()  # wgan_gp loss term
L_gdl = Gradient_Difference_Loss()  # GDL loss term

# Initialize generator and discriminator
ugan_ = PUGAN()
generator = ugan_.netG
discriminator = ugan_.netD
discriminator1 = ugan_.netD1

generator.cuda()
discriminator.cuda()
discriminator1.cuda()
L1_gp.cuda()
L1_G.cuda()
L_gdl.cuda()
Tensor = torch.cuda.FloatTensor

if (from_epoch == 0):
    generator.apply(Weights_Normal)
    discriminator.apply(Weights_Normal)
    discriminator1.apply(Weights_Normal)
else:
    generator.load_state_dict(torch.load(model_path_g_from))
    discriminator.load_state_dict(torch.load(model_path_d_from))
    discriminator1.load_state_dict(torch.load(model_path_d1_from))

optimizer_G = torch.optim.Adam(filter(lambda model: model.requires_grad, generator.parameters()), lr=lr_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate)
optimizer_D1 = torch.optim.Adam(discriminator1.parameters(), lr=lr_rate)

Dnet = DNet().cuda().eval()
Ttodnet = TtoDNet().cuda().eval()
coenet = AENet().cuda().eval()
Dnet.load_state_dict(torch.load(model_path_d))
Ttodnet.load_state_dict(torch.load(model_path_t))
coenet.load_state_dict(torch.load(model_path_a))

## Data pipeline
transforms_ = [
    transforms.Resize((img_height, img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    GetTrainingPairs(dataset_path, transforms_=transforms_),
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
)

## Training pipeline
lossD = []
lossD1 = []
lossG = []
i = 1
for epoch in range(from_epoch, num_epochs):
    for i, batch in enumerate(dataloader):
        # Model inputs
        imgs_distorted = Variable(batch["A"].type(Tensor))
        imgs_good_gt = Variable(batch["B"].type(Tensor))

        optimizer_D.zero_grad()
        with torch.no_grad():
            depth = Dnet(imgs_distorted)
            ae = coenet(imgs_distorted)
            tm = Ttodnet(ae, depth)[0]
        imgs_fake = generator(imgs_distorted, tm)
        pred_real = discriminator(imgs_good_gt)
        pred_fake = discriminator(imgs_fake)
        loss_D = -torch.mean(pred_real) + torch.mean(pred_fake)  # wgan
        gradient_penalty = L1_gp(discriminator, imgs_good_gt.data, imgs_fake.data)
        loss_D += lambda_gp * gradient_penalty  # Eq.2 paper
        loss_D.backward()
        optimizer_D.step()
        lossD.append(loss_D.item() / (len(imgs_distorted)))

        ## Train Discriminator
        optimizer_D1.zero_grad()
        with torch.no_grad():
            depth_real = torch.cat([Dnet(imgs_good_gt), imgs_good_gt], 1)
        with torch.no_grad():
            depth_fake = torch.cat([Dnet(imgs_fake), imgs_fake], 1)
        pred_real1 = discriminator1(depth_real)
        pred_fake1 = discriminator1(depth_fake)
        loss_D1 = -torch.mean(pred_real1) + torch.mean(pred_fake1)  # wgan
        gradient_penalty = L1_gp(discriminator1, depth_real.data, depth_fake.data)
        loss_D1 += lambda_gp1 * gradient_penalty  # Eq.2 paper
        loss_D1.backward()
        optimizer_D1.step()
        lossD1.append(loss_D1.item() / (len(imgs_distorted)))

        optimizer_G.zero_grad()
        ## Train Generator at 1:num_critic rate
        if i % num_critic == 0:
            with torch.no_grad():
                depth = Dnet(imgs_distorted)
                ae = coenet(imgs_distorted)
                tm = Ttodnet(ae, depth)[0]
            imgs_fake = generator(imgs_distorted, tm)
            pred_fake = discriminator(imgs_fake.detach())
            loss_gen = -torch.mean(pred_fake)
            with torch.no_grad():
                depth_fake = torch.cat([Dnet(imgs_fake), imgs_fake], 1)
            pred_fake1 = discriminator1(depth_fake.detach())
            loss_gen1 = -torch.mean(pred_fake1)
            loss_1 = L1_G(imgs_fake, imgs_good_gt)
            loss_gdl = L_gdl(imgs_fake, imgs_good_gt)
            loss_G = lambda_d * loss_gen + lambda_d1 * loss_gen1 + lambda_1 * loss_1 + lambda_2 * loss_gdl
            loss_G.backward()
            optimizer_G.step()
            lossG.append(loss_G.item() / (len(imgs_distorted)))

        ## Print log
        if not i % 50:
            sys.stdout.write("\r[Epoch %d/%d: batch %d/%d] [DLoss: %.3f, D1Loss: %.3f, 1Loss: %.3f, gdlLoss: %.3f, GLoss: %.3f] "
                             % (epoch, num_epochs, i, len(dataloader),
                                 loss_D.item(), loss_D1.item(),loss_1.item(), loss_gdl.item(), loss_G.item()
                             ))

    ## Save model checkpoints
    if ((epoch + 1) % ckpt_interval == 0):
        torch.save(generator.state_dict(),"./checkpoints/UIEB/ugan_generator_" + str(version) + str(epoch) + ".pth")
        torch.save(discriminator.state_dict(),"./checkpoints/UIEB/ugan_discriminator_" + str(version) + str(epoch) + ".pth")
        torch.save(discriminator1.state_dict(),"./checkpoints/UIEB/ugan_discriminator1_" + str(version) + str(epoch) + ".pth")
