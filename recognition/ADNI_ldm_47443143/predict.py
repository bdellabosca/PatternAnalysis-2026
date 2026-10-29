"""
predict.py
Authored by: Benigno Della Bosca

"""
import torch
import matplotlib.pyplot as plt

from modules import LDMVAE, LDMUNet

#------------------------------------------------------------------------------
SCALE = 0.82

def rev_diff(unet, vae, dev):
    with torch.no_grad():
        z = torch.randn((1,4,32,32), device=dev)
        c = torch.tensor([0], device=dev, dtype=torch.long)
        step = 1000

        for step in reversed(range(1000)):
            t = torch.tensor([step], device=dev, dtype=torch.long)

            a_bar_t = 1.0 - (step / 1000.0)
            a_bar_prev = 1.0 - ((step - 1) / 1000.0) if step > 0 else 1.0
            a_t = a_bar_t / a_bar_prev
            beta_t = 1.0 - a_t

            eps = unet(z, t, c)

            mu = (1.0 / (a_t ** 0.5)) * (z - (beta_t / ((1.0 - a_bar_t + 1e-8) ** 0.5)) * eps)

            if (step > 0):
                var = ((1.0 - a_bar_prev) / (1.0 - a_bar_t + 1e-8)) * beta_t
                sigma = var ** 0.5
                z = mu + sigma * torch.randn_like(z)
            else:
                z = mu

        synth = vae.decode(z / SCALE).clamp(0.0, 1.0).squeeze().cpu().numpy()
    return synth


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    vae = LDMVAE(in_c=1, latent_c=4).to(device)
    vae.load_state_dict(torch.load("vae_weights.pt", map_location=device))
    vae.eval()

    unet = LDMUNet(in_c=4, out_c=4, cond=256, timesteps=1000).to(device)
    unet.load_state_dict(torch.load("unet_weights.pt", map_location=device))
    unet.eval()

    img = rev_diff(unet=unet, vae=vae, dev=device)

    plt.imsave("synthetic.png", img, cmap="gray")
if __name__ == "__main__":
    main()
