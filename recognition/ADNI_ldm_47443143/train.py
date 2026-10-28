"""
train.py
Authored by: Benigno Della Bosca

Training loop for the VAE and UNet components of the LDM, the VAE training may
be perfomed distinctly, but the UNet requires a healthy latent space from a 
pre-existing VAE before training
"""
import sys
import time
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.image import StructuralSimilarityIndexMeasure

from modules import LDMVAE, LDMUNet
from dataset import get_train_ldr, get_latents_ldr

# Hyperparameters
#-------------------------------------------------------------------------------
latent_c = 4
batch_size = 32

vae_lr = 1e-4
vae_epochs = 40
vae_alpha = 0.8
vae_beta = 1e-5

unet_epochs = 150
unet_lr = 2e-4

# Loss functions
#------------------------------------------------------------------------------- 

class VAELoss(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ssim = StructuralSimilarityIndexMeasure()
        self.mae = nn.L1Loss()

    def forward(self, x_, x, mu, logvar):
        ssim_loss = 1 - self.ssim(x_, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        mae_loss = self.mae(x_, x)

        recon_loss = self.alpha * mae_loss + (1 - self.alpha) * ssim_loss
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, mae_loss, kl_loss, ssim_loss

class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, noise, expected_noise):
        return self.mse(noise, expected_noise)

# Training loop
#-------------------------------------------------------------------------------

def train_vae(model, ldr, loss, optim, dev, epochs, sched):
    model.train()
    start_time = time.time()
    
    with open("vae_loss.csv", "w", encoding="utf-8") as file:
        file.write("epoch,total_loss,mae_loss,kl_loss,ssim_loss\n")

    for epoch in range(epochs):
        epoch_total, epoch_l1, epoch_kl, epoch_ssim = 0.0, 0.0, 0.0, 0.0

        for mri, _ in ldr:
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

        n_batches = len(ldr)
        avg_total = epoch_total / n_batches
        avg_l1 = epoch_l1 / n_batches
        avg_kl = epoch_kl / n_batches
        avg_ssim = epoch_ssim / n_batches

        elapsed = time.time() - start_time
        timef = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        print(f"Epoch [{epoch+1}/{epochs}] {timef} | Loss: {avg_total:.4f} (L1:"
              f"{avg_l1:.4f}, SSIM: {avg_ssim:.4f}, KL: {avg_kl:.4f})", flush=True)

        with open("loss.csv", "a", encoding="utf-8") as file:
            file.write(f"{epoch+1},{avg_total:.6f},{avg_l1:.6f},"
                f"{avg_kl:.6f},{avg_ssim:.6f}\n")

        sched.step()

def train_unet(model, ldr, loss, optim, dev, epochs, sched):
    model.train()
    start_time = time.time()

    with open("unet_loss.csv", "w", encoding="utf-8") as file:
        file.write("epoch,loss\n")

    for epoch in range(epochs):
        total_loss = 0.0
        for latents, classes in ldr:
            latents = latents.to(dev)
            classes = classes.to(dev)

            # random 1000 t step tensor
            t = torch.randint(0, 1000, (latents.size(0), 1), device=dev)

            noise = torch.randn_like(latents)

            # make latent noisy
            s = (t.float() / 1000.0).view(-1, 1, 1, 1)
            latent_noise = torch.sqrt(1.0 - s) * latents + torch.sqrt(s) * noise

            noise_guess = model(latent_noise, t, classes) 
            mse_loss = loss(noise_guess, noise)

            optim.zero_grad()
            mse_loss.backward()
            optim.step()

            total_loss += mse_loss.item()

        n_batches = len(ldr)
        avg_loss = total_loss / n_batches

        elapsed = time.time() - start_time
        timef = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        print(f"Epoch [{epoch+1}/{epochs}] {timef} | Loss: {avg_loss:.4f}")

        with open("unet_loss.csv", "a", encoding="utf-8") as file:
            file.write(f"{epoch+1}, {avg_loss:.6f}\n")

        sched.step()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")
    
    train_ldr = get_train_ldr(sys.argv[1], batch_size=batch_size) # takes path as argument

    if (sys.argv[2] == "--vae"):

        model = LDMVAE(in_c=1, latent_c=latent_c).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=vae_lr)
        scheduler = CosineAnnealingLR(optimiser, T_max=vae_epochs, eta_min=1e-6)
        loss = VAELoss(vae_alpha, vae_beta).to(device)

        train_vae(
            model=model, 
            ldr=train_ldr, 
            loss=loss, 
            optim=optimiser, 
            dev=device, 
            epochs=vae_epochs,
            sched=scheduler
        )

        torch.save(model.state_dict(), 'vae_weights.pt')

    elif (sys.argv[2] == "--unet"):
        vae = LDMVAE(in_c=1, latent_c=4).to(device)
        vae.load_state_dict(torch.load("vae_weights.pt", map_location=device))
        vae.eval()

        latent_ldr = get_latents_ldr(vae, train_ldr, dev=device, batch_size=batch_size)

        model = LDMUNet().to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=unet_lr)
        scheduler = CosineAnnealingLR(optimiser, T_max=unet_epochs, eta_min=1e-6)
        loss = UNetLoss().to(device)

        train_unet(
            model=model,
            ldr=latent_ldr,
            loss=loss,
            optim=optimiser,
            dev=device,
            epochs=unet_epochs,
            sched=scheduler
        )

        torch.save(model.state_dict(), 'unet_weights.pt')

if __name__ == "__main__":
    main()