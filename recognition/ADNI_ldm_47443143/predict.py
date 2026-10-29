"""
predict.py
Authored by: Benigno Della Bosca

Prediction and generation script for synthesized ADNI sagittal brain MRI slices.
Supports generating samples using either the LDM or DCGAN baseline.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from modules import LDMVAE, LDMUNet
from gan import Generator

# -----------------------------------------------------------------------------

MODEL_TYPE = "GAN"
SCALE = 0.82
Z_DIM = 256

# -----------------------------------------------------------------------------

def rev_diff(unet, vae, dev):
    with torch.no_grad():
        z = torch.randn((1, 4, 32, 32), device=dev)
        c = torch.tensor([0], device=dev, dtype=torch.long)

        for step in reversed(range(1000)):
            t = torch.tensor([step], device=dev, dtype=torch.long)

            a_bar_t = 1.0 - (step / 1000.0)
            a_bar_prev = 1.0 - ((step - 1) / 1000.0) if step > 0 else 1.0
            a_t = a_bar_t / a_bar_prev
            beta_t = 1.0 - a_t

            eps = unet(z, t, c)

            mu = (1.0 / (a_t ** 0.5)) * (z - (beta_t / ((1.0 - a_bar_t + 1e-8) ** 0.5)) * eps)

            if step > 0:
                var = ((1.0 - a_bar_prev) / (1.0 - a_bar_t + 1e-8)) * beta_t
                sigma = var ** 0.5
                z = mu + sigma * torch.randn_like(z)
            else:
                z = mu

        synth = vae.decode(z / SCALE).clamp(0.0, 1.0).squeeze().cpu().numpy()
    return synth

def gan_sample(gen, dev, z_dim=Z_DIM):
    with torch.no_grad():
        noise = torch.randn(1, z_dim, 1, 1, device=dev)
        fake = gen(noise)
        synth = ((fake + 1.0) / 2.0).clamp(0.0, 1.0).squeeze().cpu().numpy()
    return synth


def generate_single_sample(dev, model_type=MODEL_TYPE):
    if model_type.upper() == "LDM":
        vae = LDMVAE(in_c=1, latent_c=4).to(dev)
        vae.load_state_dict(torch.load("vae_weights.pt", map_location=dev))
        vae.eval()

        unet = LDMUNet(in_c=4, out_c=4, cond=256, timesteps=1000).to(dev)
        unet.load_state_dict(torch.load("unet_weights.pt", map_location=dev))
        unet.eval()

        return rev_diff(unet=unet, vae=vae, dev=dev)

    elif model_type.upper() == "GAN":
        gen = Generator(z_dim=Z_DIM).to(dev)
        gen.load_state_dict(torch.load("generator_weights.pt", map_location=dev))
        gen.eval()

        return gan_sample(gen=gen, dev=dev, z_dim=Z_DIM)

def generate_grid(dev, model_type=MODEL_TYPE, rows=4, cols=4):
    grid_rows = []

    if model_type.upper() == "LDM":
        vae = LDMVAE(in_c=1, latent_c=4).to(dev)
        vae.load_state_dict(torch.load("vae_weights.pt", map_location=dev))
        vae.eval()

        unet = LDMUNet(in_c=4, out_c=4, cond=256, timesteps=1000).to(dev)
        unet.load_state_dict(torch.load("unet_weights.pt", map_location=dev))
        unet.eval()

        for r in range(rows):
            row_imgs = []
            for c in range(cols):
                img = rev_diff(unet=unet, vae=vae, dev=dev)
                row_imgs.append(img)
            grid_rows.append(np.concatenate(row_imgs, axis=1))

    elif model_type.upper() == "GAN":
        gen = Generator(z_dim=Z_DIM).to(dev)
        gen.load_state_dict(torch.load("generator_weights.pt", map_location=dev))
        gen.eval()

        for r in range(rows):
            row_imgs = []
            for c in range(cols):
                img = gan_sample(gen=gen, dev=dev, z_dim=Z_DIM)
                row_imgs.append(img)
            grid_rows.append(np.concatenate(row_imgs, axis=1))

    grid = np.concatenate(grid_rows, axis=0)
    return grid

# -----------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device} | Model :: {MODEL_TYPE}")

    grid_img = generate_grid(model_type=MODEL_TYPE, dev=device, rows=4, cols=4)
    out_grid_filename = f"{MODEL_TYPE.lower()}_grid_4x4.png"
    plt.imsave(out_grid_filename, grid_img, cmap="gray")

    """
    single_img = generate_single_sample(model_type=MODEL_TYPE, dev=device)
    out_single_filename = f"{MODEL_TYPE.lower()}_single.png"
    plt.imsave(out_single_filename, single_img, cmap="gray")
    """

if __name__ == "__main__":
    main()
