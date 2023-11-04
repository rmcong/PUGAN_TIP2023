"""
 > Script for testing .pth models
    * set model_name ('funiegan'/'ugan') and  model path
    * set data_dir (input) and sample_dir (output)
"""
# py libs
import os
import time
import argparse
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from model import DNet, TNet, TtoDNet, AENet
import numpy as np


from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

transform2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])
## options
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/input_test/")
parser.add_argument("--sample_dir", type=str, default="./sample/")
parser.add_argument("--model_path_d", type=str, default="./model/Dnet_100.pth")
parser.add_argument("--model_path_t", type=str, default="./model/Tnet_100.pth")
parser.add_argument("--model_path_a", type=str, default="./model/coe_40.pth")
opt = parser.parse_args()

## checks
assert exists(opt.model_path_d), "model not found"
assert exists(opt.model_path_t), "model not found"
if not os.path.isfile(opt.sample_dir):
    os.makedirs(opt.sample_dir, exist_ok=True)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

Dnet = DNet().cuda().eval()
Ttodnet = TtoDNet().cuda().eval()
Dnet.load_state_dict(torch.load(opt.model_path_d))
Ttodnet.load_state_dict(torch.load(opt.model_path_t))
coenet = AENet().cuda().eval()
coenet.load_state_dict(torch.load(opt.model_path_a))

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.convert('L').save(filename)

## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)

## testing loop
times = []
test_files = sorted(glob(join(opt.data_dir, "*.*")))
for path in test_files:
    inp_img = transform(Image.open(path))
    inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
    s = time.time()
    x = Dnet(inp_img)
    ae = coenet(inp_img)
    x2 = Ttodnet(ae, x)
    times.append(time.time()-s)
    save_image(x2[0], join(opt.sample_dir, basename(path)), normalize=True)
    print ("Tested: %s" % path)


## run-time
if (len(times) > 1):
    print ("\nTotal samples: %d" % len(test_files))
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print ("Time taken: %d sec at %0.3f fps" %(Ttime, 1./Mtime))
    print("Saved generated images in in %s\n" %(opt.sample_dir))



