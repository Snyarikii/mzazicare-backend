import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

class UEBAAutoencoder(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        # Compression
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3) # Latent bottleneck
        )
        # Reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def run_experiment(epochs=20, patience=3, save_dir="/content/drive/MyDrive/MzaziCare_ML_Experiments/exp1"):
    os.makedirs(save_dir, exist_ok=True)
    # [Load your CERT/synthetic telemetry dataset here]
    # ...
    # Training Loop with Early Stopping Patience and Top-3 Checkpoints:
    best_losses = []
    patience_counter = 0
    best_loss = float("inf")

    # Trains over epochs and saves checkpoints to Google Drive