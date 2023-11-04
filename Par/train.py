import os
import time
import glob
import torch
import random
import imageio
import numpy as np
from math import exp
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import skimage.transform
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import DNet, TNet, TtoDNet, AENet
from torchvision.utils import save_image


def imshow(tensor, title=None):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated
    plt.show()

def read_and_resize(paths, res=(256, 256), mode_='RGB'):
    img = imageio.imread(paths, pilmode=mode_).astype(np.float32)
    img = skimage.transform.resize(img, res)
    return img

#正则化
def noramlization(x):
    minVals = torch.min(x)
    maxVals = torch.max(x)
    ranges = maxVals - minVals
    x=(x-minVals)/ranges
    return x
#归一化
def prenarrow(x):
    return (x/255)

#读取数据集
class GetTrainingPairs(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets
    """

    def __init__(self, root, transforms1_= None, transforms2_ = None):
        self.transform1 = transforms.Compose(transforms1_)
        self.transform2 = transforms.Compose(transforms2_)
        self.filesA, self.filesB = self.get_file_paths(root)
        self.len = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.len])
        img_B = Image.open(self.filesB[index % self.len])
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B))
        img_A = self.transform1(img_A)
        img_B = self.transform2(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.len

    def get_file_paths(self, root):
        filesA = sorted(glob.glob(os.path.join(root, 'input_train') + "/*.*"))
        filesB = sorted(glob.glob(os.path.join(root, 'depth_train') + "/*.*"))

        return filesA, filesB

#退化图像
transforms1_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

#深度图gt
transforms2_ = [
    transforms.Resize((256, 256)),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize([0.5,],[0.5,])
]

dataloader = DataLoader(
    GetTrainingPairs('../data', transforms1_=transforms1_, transforms2_=transforms2_),
    batch_size = 2,
    shuffle = True,
    num_workers = 8,
)

#定义各种参数
num_epochs = 100
lr_rate = 0.001
L1_G1  = torch.nn.L1Loss() # l1 loss term
L1_G2  = torch.nn.MSELoss()
Dnet = DNet().cuda()
Ttodnet = TtoDNet().cuda()
optimizer_D = torch.optim.Adam(Dnet.parameters(), lr=lr_rate,weight_decay=0.001)
optimizer_T = torch.optim.Adam(Ttodnet.parameters(), lr=lr_rate,weight_decay=0.001)
coenet = AENet().cuda().eval()
coenet.load_state_dict(torch.load("./model/coe_40.pth"))

def train_main(epoch=0):
    print("depth_model loaded")
    print("OptimizerD loaded")
    Loss_depth = []
    Loss_depth1 = []
    for epoch in range(epoch, num_epochs):
        print('epoch {}'.format(epoch + 1))
        time_start = time.time()
        Dnet.train()
        Ttodnet.train()
        loss_depth = 0.
        loss_depth1 = 0.
        N = 0.
        for i, batch in enumerate(dataloader):
            batch_x = Variable(batch["A"]).cuda()
            batch_y = Variable(batch["B"]).cuda()
            B = batch_x.size(0)

            optimizer_D.zero_grad()
            x = Dnet(batch_x)
            loss1 = L1_G1(x,batch_y)
            loss2 = L1_G2(x, batch_y)
            loss12 = 5 * loss1 + loss2
            N += B
            loss_depth = loss_depth + loss12.item()
            loss12.backward(retain_graph=True)
            optimizer_D.step()

            ae = coenet(batch_x)
            optimizer_T.zero_grad()
            x2 = Ttodnet(ae, x.detach())
            loss4 = L1_G1(x2[1], batch_y)
            loss5 = L1_G2(x2[1], batch_y)
            loss45 = 5 * loss4 + loss5
            loss_depth1 = loss_depth1 + loss45.item()
            loss45.requires_grad_(True)
            loss45.backward()
            optimizer_T.step()

        Loss_depth.append(loss_depth / N*(len(batch_x)))
        Loss_depth1.append(loss_depth1 / N * (len(batch_x)))
        print('Depth1 Loss: {:.6f}'.format(loss_depth / N*len(batch_x)))
        print('Depth2 Loss: {:.6f}'.format(loss_depth1 / N * len(batch_x)))

        time_end = time.time()
        print('totally cost', time_end - time_start)
        if (epoch + 1) % 10 == 0:
            torch.save(Dnet.state_dict(),"./model/Dnet_" + str(epoch + 1) + ".pth")
            torch.save(Ttodnet.state_dict(),"./model/Tnet_" + str(epoch + 1) + ".pth")
    print("end")
    return Loss_depth,Loss_depth1



import matplotlib.pyplot as plt
def drow_depth():
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(depth, label="Train depth")
    plt.plot(depth2, label="Train depth1")
    plt.legend()
    plt.show()
    plt.savefig("loss_depth.png")


def drow_ae():
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(Lossae, label="train ae")
    plt.legend()
    plt.show()
    plt.savefig("loss_ae.png")

depth, depth2 = train_main()
drow_depth()
print("ok")


