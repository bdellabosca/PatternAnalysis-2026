"""
diff_train.py
Authored by: Benigno Della Bosca

Provides a training loop for the unet diffusion training componenet of the ldm,
requires a healthy trained VAE to produce latent space encodings before training
"""
import sys
import time
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import CosineAnnealingLR
from modules import LDMUNet, LDMVAE
from adni_ldr import get_train_ldr, get_latents_ldr

# Hyperparameters
#-------------------------------------------------------------------------------
batch_size = 32
lr = 2e-4
epochs = 50
#-------------------------------------------------------------------------------

class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, noise, expected_noise):
        return self.mse(noise, expected_noise)

#-------------------------------------------------------------------------------
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

    vae = LDMVAE(in_c=1, latent_c=4).to(device)
    vae.load_state_dict(torch.load("vae_weights.pt", map_location=device))
    vae.eval()

    train_ldr = get_train_ldr(sys.argv[1], batch_size=batch_size)
    latent_ldr = get_latents_ldr(vae, train_ldr, dev=device, batch_size=batch_size)

    model = LDMUNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)
    loss = UNetLoss().to(device)

    train_unet(
        model=model,
        ldr=latent_ldr,
        loss=loss,
        optim=optimiser,
        dev=device,
        epochs=epochs,
        sched=scheduler
    )

    torch.save(model.state_dict(), 'unet_weights.pt')

if __name__ == "__main__":
    main()