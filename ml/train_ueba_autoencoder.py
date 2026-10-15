%%writefile ml/train_ueba_autoencoder.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 1. Model Architecture: Deep Autoencoder for UEBA Anomaly Detection
# ---------------------------------------------------------------------------
class UEBAAutoencoder(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        # Compression (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)  # Bottleneck latent space
        )
        # Reconstruction (Decoder)
        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ---------------------------------------------------------------------------
# 2. Checkpoint Manager: Keeps and Saves Only the 3 Best Checkpoints
# ---------------------------------------------------------------------------
class Top3Checkpointer:
    def __init__(self, save_dir, max_keep=3):
        self.save_dir = save_dir
        self.max_keep = max_keep
        self.top_models = []  # List of tuples: (val_loss, epoch, filepath)

    def update(self, val_loss, epoch, model_state, optimizer_state):
        filename = f"checkpoint_epoch_{epoch}_loss_{val_loss:.4f}.pt"
        filepath = os.path.join(self.save_dir, filename)

        if len(self.top_models) < self.max_keep or val_loss < self.top_models[-1][0]:
            payload = {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state
            }
            torch.save(payload, filepath)
            self.top_models.append((val_loss, epoch, filepath))
            # Sort by validation loss ascending (lowest loss first)
            self.top_models.sort(key=lambda x: x[0])

            # Remove worst checkpoint if we exceed max_keep
            if len(self.top_models) > self.max_keep:
                worst = self.top_models.pop()
                if os.path.exists(worst[2]):
                    os.remove(worst[2])

            # Maintain clear standard symlinks/copies: best_1, best_2, best_3
            for rank, (loss, ep, path) in enumerate(self.top_models, start=1):
                best_link = os.path.join(self.save_dir, f"checkpoint_best_{rank}.pt")
                torch.save(payload, best_link)
            
            # Save overall best model
            torch.save(self.top_models[0][2], os.path.join(self.save_dir, "best_model.pt"))
            print(f" -> Checkpoint saved: Rank #{len(self.top_models)} (Val Loss: {val_loss:.5f})")

# ---------------------------------------------------------------------------
# 3. Telemetry Dataset Generation / Loading
# ---------------------------------------------------------------------------
def get_telemetry_data(csv_path=None, num_samples=1500):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        feature_cols = ['records_accessed_count', 'total_payload_bytes', 
                        'session_duration_sec', 'failed_auth_count', 'access_hour']
        data = df[feature_cols].values.astype(np.float32)
    else:
        # Generate baseline behavioral telemetry for caregiver sessions
        np.random.seed(42)
        records = np.random.poisson(lam=4, size=(num_samples, 1))
        bytes_transferred = records * np.random.normal(loc=1800, scale=300, size=(num_samples, 1)) + 500
        duration = np.random.gamma(shape=2, scale=300, size=(num_samples, 1))
        failed_logins = np.random.choice([0, 1], size=(num_samples, 1), p=[0.95, 0.05])
        hours = np.random.choice(range(7, 19), size=(num_samples, 1)) # Day shift hours

        data = np.hstack([records, bytes_transferred, duration, failed_logins, hours]).astype(np.float32)

    # Min-max normalization for neural network stability
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    denom = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
    norm_data = (data - min_vals) / denom

    split = int(0.8 * len(norm_data))
    train_data = norm_data[:split]
    val_data = norm_data[split:]

    return train_data, val_data

# ---------------------------------------------------------------------------
# 4. Training Loop with Epochs and Early Stopping Patience
# ---------------------------------------------------------------------------
def run_experiment(epochs=20, patience=3, exp_name="exp1", save_dir=None):
    if not save_dir:
        save_dir = f"/content/drive/MyDrive/MzaziCare_ML_Experiments/{exp_name}"

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f"Starting Experiment: {exp_name}")
    print(f"Max Epochs: {epochs} | Patience: {patience}")
    print(f"Save Destination: {save_dir}")
    print(f"=======================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    train_data, val_data = get_telemetry_data()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_data)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_data)), batch_size=32, shuffle=False)

    model = UEBAAutoencoder(input_dim=train_data.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    checkpointer = Top3Checkpointer(save_dir=save_dir, max_keep=3)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_train_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            batch_train_loss += loss.item() * len(batch_x)

        epoch_train_loss = batch_train_loss / len(train_data)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        batch_val_loss = 0.0
        with torch.no_grad():
            for (batch_val_x,) in val_loader:
                batch_val_x = batch_val_x.to(device)
                recon_val = model(batch_val_x)
                val_loss = criterion(recon_val, batch_val_x)
                batch_val_loss += val_loss.item() * len(batch_val_x)

        epoch_val_loss = batch_val_loss / len(val_data)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {epoch_train_loss:.5f} | Val Loss: {epoch_val_loss:.5f}", end="")

        # Update Top-3 Checkpoint manager
        checkpointer.update(epoch_val_loss, epoch, model.state_dict(), optimizer.state_dict())

        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f" -> EarlyStopping counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"\n[!] Early stopping triggered at Epoch {epoch} (No improvement for {patience} epochs).")
                break

    # Save metrics CSV to Google Drive
    metrics_df = pd.DataFrame({
        "epoch": list(range(1, len(train_losses) + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses
    })
    metrics_path = os.path.join(save_dir, "experiment_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics CSV to: {metrics_path}")

    # Generate and save loss curve plot to Google Drive
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Training Loss (MSE)", color="#2563eb", lw=2)
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Validation Loss (MSE)", color="#dc2626", lw=2, linestyle="--")
    plt.title(f"UEBA Autoencoder Loss: {exp_name} (Patience: {patience})")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Reconstruction Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve plot to: {plot_path}")
    print(f"\nExperiment {exp_name} finished successfully.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs to train")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--exp-name", type=str, default="exp1", help="Experiment name")
    parser.add_argument("--save-dir", type=str, default="", help="Custom Google Drive save path")
    args = parser.parse_args()

    run_experiment(epochs=args.epochs, patience=args.patience, exp_name=args.exp_name, save_dir=args.save_dir if args.save_dir else None)