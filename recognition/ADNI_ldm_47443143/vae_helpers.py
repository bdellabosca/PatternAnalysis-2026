"""
vae_helpers.py
Authored by: Benigno Della Bosca

Some Helper functions for the vae pipeline
"""
import torch
from modules import LDMVAE

def get_latents(model, ldr, dev):
    pass

def plot_umap():
    pass

def sample_inference():
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    test_ldr = get_test_ldr(sys.argv[1]) 
    model = LDMVAE(in_c=1, latent_c=4).to(device)
    state = torch.load("vae_weights.pt")
    model.load_state_dict(state)

if __name__ == "__main__":
    main()