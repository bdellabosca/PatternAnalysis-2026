"""
predict.py
Authored by: Gemini 3.8

Prediction and evaluation script for synthesized ADNI sagittal brain MRI slices.
Runs generation and evaluation across both LDM and DCGAN baselines when called:
- 5x5 Grid generation & single image output
- Diversity check (Pairwise SSIM)
- Memorization check (Nearest-Neighbor SSIM against real training slices)

Optimized with GPU-vectorized SSIM to accelerate the memorization check from minutes to seconds.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from modules import LDMVAE, LDMUNet
from gan import Generator
from dataset import ADNIDataset, nil_transform, train_path

# -----------------------------------------------------------------------------

MODELS = ["LDM", "GAN"]  # Evaluates both models sequentially
SCALE = 0.82             # Latent scale factor for LDM VAE
Z_DIM = 256              # Noise dimension for GAN
DATA_DIR = "ADNI"        # Base dataset directory

# -----------------------------------------------------------------------------

def fast_ssim_batch(img1, img2):
    """
    Computes paired SSIM between batches (B, 1, H, W) directly on GPU in a single kernel.
    Returns a 1D tensor of shape (B,).
    """
    coords = torch.arange(11, dtype=torch.float32, device=img1.device) - 5.0
    g = torch.exp(-0.5 * (coords / 1.5) ** 2)
    window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
    window = window / window.sum()

    mu1 = F.conv2d(img1, window, padding=5)
    mu2 = F.conv2d(img2, window, padding=5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2

    sig1_sq = F.conv2d(img1 * img1, window, padding=5) - mu1_sq
    sig2_sq = F.conv2d(img2 * img2, window, padding=5) - mu2_sq
    sig12 = F.conv2d(img1 * img2, window, padding=5) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2.0 * mu1_mu2 + c1) * (2.0 * sig12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sig1_sq + sig2_sq + c2))
    return ssim.mean(dim=[1, 2, 3])

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
                z = mu + (var ** 0.5) * torch.randn_like(z)
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

# -----------------------------------------------------------------------------

def generate_grid(model_type, dev, rows=5, cols=5):
    total_samples = rows * cols
    sample_list = []

    if model_type.upper() == "LDM":
        vae = LDMVAE(in_c=1, latent_c=4).to(dev)
        vae.load_state_dict(torch.load("vae_weights.pt", map_location=dev))
        vae.eval()

        unet = LDMUNet(in_c=4, out_c=4, cond=256, timesteps=1000).to(dev)
        unet.load_state_dict(torch.load("unet_weights.pt", map_location=dev))
        unet.eval()

        for idx in range(total_samples):
            img = rev_diff(unet=unet, vae=vae, dev=dev)
            sample_list.append(img)

    elif model_type.upper() == "GAN":
        gen = Generator(z_dim=Z_DIM).to(dev)
        gen.load_state_dict(torch.load("generator_weights.pt", map_location=dev))
        gen.eval()

        for idx in range(total_samples):
            img = gan_sample(gen=gen, dev=dev, z_dim=Z_DIM)
            sample_list.append(img)

    grid = np.block([[sample_list[r * cols + c] for c in range(cols)] for r in range(rows)])
    return grid, sample_list

# -----------------------------------------------------------------------------

def evaluate_model(model_type, samples, dev, train_loader=None):
    """
    Fast GPU-vectorized evaluation of:
    1. Pairwise SSIM (Diversity check across all 300 sample pairs)
    2. Nearest-Neighbor SSIM (Memorization check against real training slices)
    """
    tensors = [torch.from_numpy(s).unsqueeze(0).unsqueeze(0).float() for s in samples]
    samples_tensor = torch.cat(tensors, dim=0).to(dev)
    n_samples = samples_tensor.size(0)

    # 1. Diversity (Pairwise SSIM across all 300 pairs in a single GPU pass)
    print(f"\n[2/3] Evaluating Pairwise SSIM (Diversity) across {n_samples} samples...")
    idx_i, idx_j = torch.triu_indices(n_samples, n_samples, offset=1)
    pairwise_scores = fast_ssim_batch(samples_tensor[idx_i], samples_tensor[idx_j])
    mean_pairwise = pairwise_scores.mean().item()

    # 2. Memorization (Nearest-Neighbor SSIM against training set)
    nn_max_scores = torch.zeros(n_samples, device=dev)

    if train_loader is not None:
        total_batches = len(train_loader)
        print(f"\n[3/3] Fast scanning {total_batches} training batches on {dev} for Nearest-Neighbor SSIM...")

        with torch.no_grad():
            for batch_idx, (real_batch, _) in enumerate(train_loader):
                real_batch = real_batch.to(dev)
                b_size = real_batch.size(0)

                for i in range(n_samples):
                    sample_rep = samples_tensor[i:i + 1].expand(b_size, 1, 256, 256)
                    scores = fast_ssim_batch(sample_rep, real_batch)
                    max_in_batch = scores.max()
                    if max_in_batch > nn_max_scores[i]:
                        nn_max_scores[i] = max_in_batch

                if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == total_batches:
                    curr_max = nn_max_scores.max().item()
                    print(f"  Scanned batch [{batch_idx + 1:02d}/{total_batches:02d}] | Current Highest NN SSIM: {curr_max:.4f}")

    mean_nn = nn_max_scores.mean().item() if n_samples > 0 else 0.0
    max_nn = nn_max_scores.max().item() if n_samples > 0 else 0.0

    # Single-line output at the end of the evaluation
    print(f"\n[{model_type.upper()}] Pairwise SSIM (Diversity): {mean_pairwise:.4f} | NN SSIM (Memorization) Mean: {mean_nn:.4f} | Max: {max_nn:.4f}")
    return {"pairwise_ssim": mean_pairwise, "nn_mean": mean_nn, "nn_max": max_nn}

# -----------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    train_ds = ADNIDataset(DATA_DIR, path=train_path, transform=nil_transform)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    for model_type in MODELS:
        print("\n" + "=" * 60)
        print(f" RUNNING EVALUATION PIPELINE: {model_type}")
        print("=" * 60)

        grid_img, samples = generate_grid(model_type=model_type, dev=device, rows=5, cols=5)
        out_grid_filename = f"{model_type.lower()}_grid_5x5.png"
        plt.imsave(out_grid_filename, grid_img, cmap="gray")
        print(f"\nSaved 5x5 grid to {out_grid_filename}")

        evaluate_model(model_type=model_type, samples=samples, dev=device, train_loader=train_loader)


if __name__ == "__main__":
    main()
