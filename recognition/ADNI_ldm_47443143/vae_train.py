"""
vae_train.py
Authored by: Benigno Della Bosca

Provides a training loop for the VAE component of the ldm model, can be trained
distinct of other component architectures
"""
import torch
import torch.nn as nn

from modules import LDMVAE

# Hyperparameters
#-------------------------------------------------------------------------------
batch_size = 0
lr = 0
epochs = 0
beta = 0
#------------------------------------------------------------------------------- 

def vae_loss():
    pass

def train_vae():
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

if __name__ == "__main__":
    main()