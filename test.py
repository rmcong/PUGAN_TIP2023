import os
import time
import argparse
import yaml
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from Par.model import DNet, TtoDNet, AENet
from torchvision.transforms import InterpolationMsode


## options
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file", type=str, default="test.yaml")
args = parser.parse_args()

with open(args.cfg_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
# get info from config file
dataset_path = cfg["data_dir"]
sample_path = cfg["sample_dir"]
gtr_path = cfg["gtr_dir"]
img_width = cfg["im_width"]
img_height = cfg["im_height"]
channels = cfg["chans"]
model_path = cfg["path_gen"]
model_path_d = cfg["path_d"]
model_path_t = cfg["path_t"]
model_path_a = cfg["path_a"]

## checks
assert exists(model_path), "Generator model not found"
assert exists(model_path_a), "AENet model not found"
assert exists(model_path_d), "DNet model not found"
assert exists(model_path_t), "TNet model not found"
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
## model arch

from nets.fusion import PUGAN
model = PUGAN().netG
Dnet = DNet()
Ttodnet = TtoDNet()
coenet = AENet()

## load weights
model.load_state_dict(torch.load(model_path))
model.cuda().eval()
Dnet = DNet().cuda().eval()
Ttodnet = TtoDNet().cuda().eval()
coenet = AENet().cuda().eval()
Dnet.load_state_dict(torch.load(model_path_d))
Ttodnet.load_state_dict(torch.load(model_path_t))
coenet.load_state_dict(torch.load(model_path_a))

## data pipeline
transforms_ = [transforms.Resize((img_height, img_width), InterpolationMode.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)

## testing loop
times = []
test_files = sorted(glob(join(dataset_path, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    depth = Dnet(inp_img)
    ae = coenet(inp_img)
    tm = Ttodnet(ae, depth)[0]
    s = time.time()
    gen_img = model(inp_img, tm)
    times.append(time.time()-s)
    save_image(gen_img, join(sample_path, basename(path)), normalize=True)
    print ("Tested: %s" % path)

## run-time    
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files))
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(sample_path))

with open("./evaluations/measure.txt",'a+') as f:
    f.write('test on ' + dataset_path + 'use ' + model_path + ':')
    f.write('\n')
    print("end testing")
f.close()


