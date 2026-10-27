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

    def get_latent(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return z

    def forward(self, x):
        (mu, logvar) = self.encode(x)
        x_ = self.decode(self.reparameterise(mu, logvar))
        return (x_, mu, logvar)

#-------------------------------------------------------------------------------
# UNet
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, condition):
        super().__init__()
        self.condition = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition, out_c)
        )
        self.blk1 = nn.Sequential(
            nn.GroupNorm(32, in_c),
            nn.SiLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
        )
        self.blk2 = nn.Sequential(
            nn.GroupNorm(32, out_c),
            nn.SiLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )
        self.res = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x, t):
        h = self.blk1(x)
        h = h + self.condition(t)[:, :, None, None]
        h = self.blk2(h)
        return (h + self.res(x))

class LDMUNet(nn.Module):
    def __init__(self, in_c=4, out_c=4, f=64, cond=256, timesteps=1000):
        super().__init__()
        self.time = nn.Embedding(timesteps, cond)
        self.classes = nn.Embedding(2, cond) 
            
        # needs mlp for time conditoning
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond, cond),
        )

        # from latent space to features
        self.from_latent = nn.Conv2d(in_c, f, kernel_size=3, padding=1)

        # down

        # up

        # from features back to latent
        self.to_latent = nn.Sequential(
            nn.GroupNorm(32, f),
            nn.SiLU(),
            nn.Conv2d(f, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x, tstep, classification):
        condition = self.time(tstep) + self.classes(classification)
        condition = self.mlp(condition)

        pass
        
