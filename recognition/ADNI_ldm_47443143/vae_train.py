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

class VAELoss(nn.Module):
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

        recon_loss = alpha * mae_loss + (1 - alpha) * ssim_loss
        total_loss = recon_loss + beta * kl_loss

        return total_loss, mae_loss, ssim_loss, kl_loss

#-------------------------------------------------------------------------------

def train_vae(model, ldr, loss, optim, dev, epochs):
    model.train()

    for epoch in range(epochs):
        epoch_total, epoch_l1, epoch_kl, epoch_ssim = 0.0, 0.0, 0.0, 0.0

        for mri in NotImplemented:
            mri = mri.to(dev)

            optim.zero_grad()
            mri_, mu, logvar = model(mri)
            total_l, mae_l, kl_l, ssim_l = loss(mri_, mri, mu, logvar) 

            total_l.backward()            
            optim.step()

            epoch_total += total_l.item()
            epoch_l1 += mae_l.item()
            epoch_kl += kl_l.item()
            epoch_ssim += ssim_l.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    #TODO: needs loaders

    model = LDMVAE(in_c=1, latent_c=latent_c).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss = VAELoss()

    train_vae(
        model, 
        ldr=NotImplemented, 
        loss=loss, 
        optim=optimiser, 
        dev=device, 
        epochs=epochs
    )

if __name__ == "__main__":
    main()