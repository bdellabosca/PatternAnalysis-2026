"""
vae_train.py
Authored by: Benigno Della Bosca

Provides a training loop for the VAE component of the ldm model, can be trained
distinct of other component architectures
"""
import torch
import torch.nn as nn

from torchmetrics.image import StructuralSimilarityIndexMeasure

from modules import LDMVAE

# Hyperparameters
#-------------------------------------------------------------------------------
latent_c = 4

batch_size = 0
lr = 0
epochs = 0
alpha = 0
beta = 0
#------------------------------------------------------------------------------- 

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mae = nn.L1Loss()

    def forward(self, x_, x, mu, logvar):
        ssim_loss = 1 - self.ssim(x_, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        mae_loss = self.mae(x_, x)

        # a * mae + (1 - a)(ssim_loss)
        recon_loss = NotImplemented
        # recon + b * kl
        total_loss = NotImplemented

        return total_loss, ssim_loss, kl_loss

#-------------------------------------------------------------------------------

def train_vae(model, ldr, optim, dev, epochs, beta):
    model.train()

    for epoch in range(epochs):
        #TODO: per epoch train and loss calc 
        pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    #TODO: needs loaders

    model = LDMVAE(in_c=1, latent_c=latent_c).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    train_vae(model, "", optimiser, device, epochs=epochs, beta=beta)


if __name__ == "__main__":
    main()