"""
modules.py
Authored by: Benigno Della Bosca

Implementations of VAE and UNet architectures for use in the latent diffusion
model.
"""
import torch
import torch.nn as nn

ENCODE = 0
DECODE = 1
#-------------------------------------------------------------------------------
# VAE
class VAEBlock(nn.Module):
    def __init__(self, blocktype, in_c, out_c, kern, groups=32):
        super().__init__()
        if blocktype == ENCODE:
            self.do = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=kern, stride=2, padding=1),
                nn.GroupNorm(groups, out_c),
                nn.SiLU()
            )
        elif blocktype == DECODE:
            self.do = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=kern, stride=2, 
                                   padding=1),
                nn.GroupNorm(groups, out_c),
                nn.SiLU()
            )

    def forward(self, x):
        return self.do(x)

class LDMVAE(nn.Module):
    def __init__(self, in_c=1, latent_c=4):
        super().__init__()
        self.encoder = nn.Sequential(
            VAEBlock(ENCODE, in_c, 64, 4),
            VAEBlock(ENCODE, 64, 128, 4),
            VAEBlock(ENCODE, 128, 256, 4)
        )

        self.mu = nn.Conv2d(256, latent_c, kernel_size=1, padding=0)
        self.logvar = nn.Conv2d(256, latent_c, kernel_size=1, padding=0)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_c, 256, kernel_size=3, padding=1),
            nn.SiLU(),
            VAEBlock(DECODE, 256, 128, 4),
            VAEBlock(DECODE, 128, 64, 4),
            nn.ConvTranspose2d(64, in_c, kernel_size=4, stride=2, padding=1)
        )

    def reparameterise(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma)
        return (mu + sigma * epsilon) # z

    def encode(self, x):
        hidden = self.encoder(x)
        return (self.mu(hidden), self.logvar(hidden))

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        (mu, logvar) = self.encode(x)
        x_ = self.decode(self.reparameterise(mu, logvar))
        return (x_, mu, logvar)

#-------------------------------------------------------------------------------
# UNet
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.do = nn.Sequential(
            # conv
            # norm
            # activ fn
        )

    def forward(self):
        return self.do

class LDMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # down blocks

        # conditioning mlp

        # up blocks

    def forward(self):
        # down + maintain skip conns
        # up