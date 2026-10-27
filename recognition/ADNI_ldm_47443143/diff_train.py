"""
diff_train.py
Authored by: Benigno Della Bosca

Provides a training loop for the unet diffusion training componenet of the ldm,
requires a healthy trained VAE to produce latent space encodings before training
"""
import torch
import torch.nn as nn

from modules import LDMUNet

# Hyperparameters
#-------------------------------------------------------------------------------
batch_size = 0
lr = 0
epochs = 0
#-------------------------------------------------------------------------------

class UNetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass

#-------------------------------------------------------------------------------
def train_unet(model, ldr, loss, optim, dev, epochs):
    model.train()

    for epoch in range(epochs):
        pass

    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device :: {device}")

    # TODO: implement loader -> pre encode latents for entire dataset

    model = LDMUNet()
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    loss = UNetLoss().to(device)

    train_unet(
        model=model,
        ldr=NotImplemented,
        loss=loss,
        optim=optimiser,
        dev=device,
        epochs=epochs
    )


if __name__ == "__main__":
    main()