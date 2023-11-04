import os
import xlrd
import time
import torch
import random
import imageio
import numpy as np
from math import exp
from PIL import Image
import torch.nn as nn
import skimage.transform
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import AENet

unloader = transforms.ToPILImage()
loader = transforms.Compose([transforms.ToTensor()])

#输入图像tensor   展示图像
def imshow(tensor, title=None):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it- clone深拷贝
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
        plt.pause(0.001) # pause a bit so that plots are updated
    plt.show()

    #resize图像  112*112
def read_and_resize(paths, res=(112, 112), mode_='RGB'):
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


class loaddata_test():
    def __init__(self, path):
        self.imgsize = (256, 256)
        self.folder = "data_coe/"
        self.get_paths(path)

    def get_paths(self, path):
        self.num_train, self.num_val = 0, 0
        self.train_paths, self.val_paths = [], []
        self.gt_tr_paths, self.gt_val_paths = [], []
        data_dir = os.path.join(path, self.folder)
        data_path = sorted(os.listdir(data_dir))
        num_paths = max(len(data_path), 0)
        all_idx = list(range(num_paths))
        # 95% train-val splits
        random.seed(2)
        random.shuffle(all_idx)
        self.num_val = 12
        self.num_train = num_paths - self.num_val
        train_idx = set(all_idx[:self.num_train])
        # split data paths to training and validation sets
        for i in range(num_paths):
            if i in train_idx:
                self.train_paths.append(data_dir + str(i+1) + ".jpg")
                self.gt_tr_paths.append(str(i+1))
            else:
                self.val_paths.append(data_dir + str(i) + ".jpg")
                self.gt_val_paths.append(str(i))
        print("Loaded {0} samples for training".format(self.num_train))

    def get_path(self, index):
        return self.train_paths[index], self.gt_tr_paths[index]

# 将训练集集路径转为tensor的加载器
class Dataset_test_tr(object):
    def __init__(self, path):
        self.load = loaddata_test(path)
        self.resimg = (256, 256)
        self.le = 0
        self.le = len(self.load.train_paths)
        xls_file = xlrd.open_workbook(path + "coe.xls")
        self.xls_sheet = xls_file.sheets()[0]

    # 读取img  转为tensor   读取lable  转为tensor
    def __getitem__(self, index):
        path = self.load.train_paths[index]
        pid = self.load.gt_tr_paths[index]
        img = read_and_resize(path, res=self.resimg)
        row_value = self.xls_sheet.row_values(int(pid) - 1)
        imgs_tensor = torch.from_numpy(np.transpose(prenarrow(np.array(img)), (2, 0, 1)))
        imgs_label = torch.Tensor(row_value)
        return imgs_tensor, imgs_label

    def __len__(self):
        return self.le

loss_=nn.L1Loss()

def train_coe(epoch_num=41):
    train_data=Dataset_test_tr("../data/coe_data/")
    batch_size=1
    train_loader = DataLoader(dataset=train_data, batch_size=4, shuffle=True,drop_last=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coe_model=AENet().to(device)
    print("COE_model loaded")
    optimizer = torch.optim.Adam(coe_model.parameters())
    print("Optimizer loaded")
    Loss_train = []
    for epoch in range(epoch_num):
        time_start=time.time()
        print('epoch {}'.format(epoch + 1))
        # training-----------------------------
        coe_model.train()
        train_loss = 0.
        for batch_x, batch_y in train_loader:
            batch_x=batch_x.cuda()
            batch_y=batch_y.cuda()
            out = coe_model(batch_x)
            loss = loss_(out, batch_y)
            #print(out)
            train_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        Loss_train.append(train_loss / (len(train_data))*batch_size)
        print('Train Loss: {:.6f}'.format(train_loss / (len(train_data))))
        time_end=time.time()
        print('totally cost',time_end-time_start)
        if (epoch+1)%20==0:
            torch.save(coe_model.state_dict(),"model/coe_"+str(epoch+1)+".pth")
    print("end")
    return Loss_train

import matplotlib.pyplot as plt
def drow():
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(Train_Loss, label ="Train Loss")
    plt.legend()
    plt.show()
    plt.savefig("coe_loss.png")

#开始训练并返回train和test损失
Train_Loss=train_coe()
drow()