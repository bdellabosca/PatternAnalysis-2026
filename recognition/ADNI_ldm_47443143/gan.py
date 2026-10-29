"""
gan.py
Authored by: Benigno Della Bosca

A DCGAN implementation as a baseline for comparing the overall effectiveness of
the ldm. Unlike the rest of the ldm this file will be monolithic and separate,
with the intention of being separated from  the rest of the ldm as needed.

Implementation mostly borrowed from a2
"""

import torch
import torch.nn as nn

#Generator
#-------------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, z_dim, out_c=1, f=32):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, f * 8, 4, 1, 0),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(f * 8, f * 8, 4, 2, 1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 8, f * 8, 4, 2, 1),
            nn.BatchNorm2d(f * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1),
            nn.BatchNorm2d(f * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1),
            nn.BatchNorm2d(f * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1),
            nn.BatchNorm2d(f),
            nn.ReLU(True),

            nn.ConvTranspose2d(f, out_c, 4,2,1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# Discriminator
#-------------------------------------------------------------------------------

class Discriminator(nn.Module):
    def __init__(self, in_c=1, f=32):
        super().__init__()
        self.discrim = nn.Sequential(
            nn.Conv2d(in_c, f, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f, f * 2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 4, f * 8, 4,2,1,bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 8, f * 8, 4,2,1,bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 8, f * 8, 4,2,1,bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(f * 8, 1,4,1,0, bias=False),
        )

    def forward(self, x):
        return self.discrim(x).view(-1)