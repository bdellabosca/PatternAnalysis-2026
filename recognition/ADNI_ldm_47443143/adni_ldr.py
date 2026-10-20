"""
ADNI_ldr.py
Authored by: Benigno Della Bosca

Provides an interface for loading ADNI MR images, as well as per epoch
transformations
"""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

#NOTE: Currently there is no patient level separation on dataset
# DO NOT USE FOR FINAL MODEL

# Parameters
#-------------------------------------------------------------------------------

train_path = "AD_NC/train"
test_path = "AD_NC/test"

rotation = (-5, 5)
translation = (0.05, 0.05)
brightness = 0.1
contrast = 0.1

# Transform
#-------------------------------------------------------------------------------

augment = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=rotation, fill=0),
    transforms.RandomAffine(degrees=0, translate=translation, fill=0),
    transforms.ColorJitter(brightness=brightness, contrast=contrast)
])

nil_transform = transforms.Compose([
    transforms.ToTensor()
])

#-------------------------------------------------------------------------------

class ADNIDataset(Dataset):
    def __init__(self, base, path, transform):
        self.transform = transform
        self.cn_dir =  os.path.join(base, path, "CN")
        self.ad_dir =  os.path.join(base, path, "AD")
        self.slices = []

        for file in os.listdir(self.cn_dir):
            filepath = os.path.join(self.cn_dir, file)
            self.slices.append((filepath,"CN"))

        for file in os.listdir(self.ad_dir):
            filepath = os.path.join(self.ad_dir, file)
            self.slices.append((filepath, "AD"))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, ix):
        path, classification = self.slices[ix]
        slice = Image.open(path).convert("L")
        if self.transform: slice = self.transform(slice)
        return slice, classification

#-------------------------------------------------------------------------------
def get_test_ldr(base, batch_size=32, num_workers=4):
    test_ds = ADNIDataset(base, path=test_path, transform=nil_transform)
    test_ldr = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return test_ldr 

def get_train_ldr(base, batch_size=32, num_workers=4): 
    train_ds = ADNIDataset(base, path=train_path, transform=augment)
    train_ldr = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, # random order per epoch
        num_workers=num_workers
    )
    return train_ldr