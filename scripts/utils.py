import torch
from torch.utils.data import DataLoader
from torchvision import datasets, utils, transforms
from UNet_Multiclass.scripts.datasets import SuperMarioDataset
from PIL import Image
from tqdm import tqdm
import numpy as np

def get_supermario_data(
    split,
    root_dir='datasets/superMario',
    target_type="semantic",
    transforms=None,
    batch_size=1,
    eval=False,
    shuffle=True,
    pin_memory=True,

):
    data = SuperMarioDataset(
        split=split, target_type=target_type, transform=transforms, root_dir=root_dir, eval=eval)

    data_loaded = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

    return data_loaded

# Functions to save predictions as images 
def save_as_images(tensor_pred, folder, image_name):
    tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}/{image_name}.png"
    tensor_pred.save(filename)

#Montalvo mio padre
def decode_segmap(image, nc=21):
    ## Color palette for visualization of the 21 classes
    label_colors = np.array([(0, 0, 0),  # 0=background
                             # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                             (0, 0, 255), (127, 127, 0), (0, 255, 0), (255, 0, 0), (255, 255, 0),
                             # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                             (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                             # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                             (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                             # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                             (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb

        
def decode_layer(image, nc=1024, label_colors=None):

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    label_colors = list(label_colors)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l][0]
        g[idx] = label_colors[l][1]
        b[idx] = label_colors[l][2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb
    