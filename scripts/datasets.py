import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
import os


class SuperMarioDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', mode='fine', transform=None, eval=False):
        self.transform = transform
        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval

        # Preparing a list of all labelTrainIds rgb and
        # ground truth images. Setting relabbelled=True is recommended.

        self.label_path = os.path.join(os.getcwd(), root_dir + '/' + self.split + '/' + 'lables/')
        self.rgb_path = os.path.join(os.getcwd(), root_dir + '/' + self.split + '/' + 'imgs/')
        self.XImg_list = os.listdir(self.rgb_path)
        self.yLabel_list = os.listdir(self.label_path)

    def __len__(self):
        length = len(self.XImg_list)
        return length

    def __getitem__(self, index):
        image = Image.open(self.rgb_path + self.XImg_list[index])
        y = Image.open(self.label_path + self.yLabel_list[index])

        if self.transform is not None:
            image = self.transform(image)
            y = self.transform(y)

        image = transforms.ToTensor()(image)
        y = np.array(y)
        y = torch.from_numpy(y)

        y = y.type(torch.LongTensor)
        if self.eval:
            return image, y, self.XImg_list[index]
        else:
            return image, y