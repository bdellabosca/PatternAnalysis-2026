"""
modules.py
Authored by: Benigno Della Bosca

Implementations of VAE and UNet architectures for use in the latent diffusion
model.
"""
import torch
import torch.nn as nn

#-------------------------------------------------------------------------------
# VAE 
class LDMVAE(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.encoder = nn.Sequential()

        self.mu = nn.Conv2d()
        self.logvar = nn.Conv2d()
        
        self.decoder = nn.Sequential()

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
# TODO:  Implement UNet architecture