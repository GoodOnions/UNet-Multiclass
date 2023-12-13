import numpy as np
import torch
import utils
import matplotlib.pyplot as plt
from utils import get_supermario_data, decode_segmap
import torchvision.transforms as transforms
from model import UNET
import os

os.chdir('/Users/daniele/KTH-Projects/Semantic-Segmentation-for-Deep-Reinforcement-Learning/UNet_Multiclass/scripts')

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
    print('Running on the GPU')
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    print('Running on the MPS')
else:
    DEVICE = "cpu"
    print('Running on the CPU')

# Dataset settings
ROOT_DIR = '../datasets/superMario'
IMG_HEIGHT = 240
IMG_WIDTH = 272

# Model
path = '../models/e60_b32_grey_unet_super.pth'
IN_CHANNELS = 1
CLASSES = 6


test_set = get_supermario_data(
        split='test',
        root_dir=ROOT_DIR,
        transforms=None,
        batch_size=1,
    )

net = UNET(in_channels=IN_CHANNELS, classes=CLASSES).to(DEVICE)
checkpoint = torch.load(path, map_location=DEVICE)
net.load_state_dict(checkpoint['model_state_dict'])

# Get random sample from the dataset
sample = next(iter(test_set))

prediction = net(sample[0].to(DEVICE))
prediction = torch.nn.functional.softmax(prediction, dim=1)
prediction = torch.argmax(prediction, dim=1).squeeze()
prediction = prediction.float().detach().cpu().numpy()
segm_rgb = decode_segmap(prediction)

# from tensor to img
img = transforms.ToPILImage()(sample[0].squeeze().cpu())
segm_rgb = net.predict(img)
plt.imshow(segm_rgb)
plt.axis('off')
plt.savefig('../predictions/test.png', format='png',dpi=300,bbox_inches = "tight")