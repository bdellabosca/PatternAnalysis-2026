"""
diff_train.py
Authored by: Benigno Della Bosca

Provides a training loop for the unet diffusion training componenet of the ldm,
requires a healthy trained VAE to produce latent space encodings before training
"""
import torch
import torch.nn as nn

from modules import LDMUNet, LDMVAE
from adni_ldr import get_train_ldr, get_latents_ldr

# Hyperparameters
#-------------------------------------------------------------------------------
batch_size = 0
lr = 0
epochs = 0
#-------------------------------------------------------------------------------

class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, noise, expected_noise):
        return self.mse(noise, expected_noise)

#-------------------------------------------------------------------------------
def train_unet(model, ldr, loss, optim, dev, epochs):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for latents, classes in ldr:
            latents = latents.to(dev)
            classes = classes.to(dev)

            # random 1000 t step tensor
            t = torch.randint(0, 1000, (latents.size(0), 1), device=dev)

            noise = torch.randn_like(latents)

            # make latent noisy

            noise_guess = model(NotImplemented, t, classes) 
            mse_loss = loss(noise_guess, noise)

            optim.zero_grad()
            mse_loss.backward()
            optim.step()

            total_loss += mse_loss.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    model = LDMUNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = UNetLoss().to(device)

    train_unet(
        model=model,
        ldr=get_latents_ldr,
        loss=loss,
        optim=optimiser,
        dev=device,
        epochs=epochs
    )


if __name__ == "__main__":
    main()