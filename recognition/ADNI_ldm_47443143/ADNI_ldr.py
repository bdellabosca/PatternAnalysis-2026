"""
ADNI_ldr.py
Authored by: Benigno Della Bosca

Provides an interface for loading ADNI MR images, as well as per epoch
transformations
"""

import os
import re
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#NOTE: Currently there is no patient level separation on dataset
# DO NOT USE FOR FINAL MODEL

# Paths
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

class ADNIDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, ix):
        pass

#-------------------------------------------------------------------------------

def get_test_ldr():
    pass

def get_train_ldr(): 
    pass