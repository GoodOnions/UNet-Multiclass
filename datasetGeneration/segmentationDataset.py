import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_folder = os.path.join(self.root_dir, 'PNG')
        self.masks_folder = os.path.join(self.root_dir, 'Labels')
        self.images = os.listdir(self.images_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_folder, self.images[idx])
        mask_name = os.path.join(self.masks_folder, self.images[idx])  # Assuming same filenames for images and masks

        image = Image.open(img_name)#.convert('RGB')
        mask = Image.open(mask_name)#.convert('L')  # Convert mask to grayscale if necessary

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
    


