"""
SC2 Protoss Imitation Learning — MLP Model + Training Script
=============================================================
Architecture: 31 -> 256 -> 256 -> 128 -> 30
Activations:  GELU  (ReLu but avoids dying neurons)
Regularization: Dropout(0.3) between hidden layers (some features are related)
Loss: CrossEntropyLoss with class weights (handles imbalanced action distribution)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — edit these paths to match your setup
# ---------------------------------------------------------------------------
DATASET_PATH = r"C:\dev\BetaStar\replays\parsed\dataset.npz"
CHECKPOINT_DIR = r"C:\dev\BetaStar\checkpoints"


OBS_SIZE = 30  # from observation_wrapper.py input length
NUM_ACTIONS = 30  # from actions.py action length

BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4   # Adam's "golden" LR for MLPs
VAL_SPLIT = 0.15
DROPOUT = 0.3
SEED = 54

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ProtossModel(nn.Module):
    """
    3-hidden-layer MLP for macro action prediction.

    Why this architecture:
    - 256->256->128: wider early layers capture feature interactions
      (e.g. minerals + supply + structure counts), narrower final layer
      funnels into 30 action logits without forcing too abrupt a collapse.
    - GELU: smooth activation that keeps gradients alive through all neurons.
      ReLU can permanently zero out neurons ("dying ReLU"); GELU avoids this.
    - Dropout(0.3): obs vector has correlated features (supply_used,
      supply_cap, supply_left all move together). Dropout prevents the network
      from over-relying on any single feature.
    - BatchNorm before activation: stabilizes training on the unnormalized
      resource values that slip through despite your /1800 normalization.
    """

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        num_actions: int = NUM_ACTIONS,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        def block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.net = nn.Sequential(
            block(obs_size, 256),
            block(256, 256),
            block(256, 128),
            # straight logits — CrossEntropyLoss handles softmax
            nn.Linear(128, num_actions),
        )

        # Weight init: Kaiming (He) is designed for ReLU-family activations incl. GELU.
        # It keeps variance stable as signal passes through deep layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Inverse-frequency weighting so rare actions (e.g. build_fleet_beacon)
    get the same gradient signal as common ones (train_probe, build_pylon).
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    # avoid div/0 for unseen actions
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights /= weights.sum()                        # normalize to sum=1
    return torch.tensor(weights, dtype=torch.float32)


def load_dataset(path: str):
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    print(
        f"Action distribution: { {int(i): int((data['y']==i).sum()) for i in np.unique(data['y'])} }")
    return X, y


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Device ---
    # CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU found)")

    # --- Data ---
    X, y = load_dataset(DATASET_PATH)
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {train_size} samples | Val: {val_size} samples")

    # --- Model ---
    model = ProtossModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # --- Loss with class weights ---
    class_weights = compute_class_weights(y.numpy(), NUM_ACTIONS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Optimizer + Scheduler ---
    # AdamW is Adam with proper weight decay (decoupled from gradient updates).
    # Weight decay acts as L2 regularization — discourages large weights.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # CosineAnnealingLR smoothly decays LR to near-zero over training.
    # Better than step decay because it avoids abrupt LR drops that can destabilize training.
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- Checkpointing ---
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_path = Path(CHECKPOINT_DIR) / "best_model.pt"

    # --- Training loop ---
    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>10.3%} {val_loss:>10.4f} {val_acc:>9.3%} {current_lr:>10.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "obs_size": OBS_SIZE,
                "num_actions": NUM_ACTIONS,
            }, best_path)
            print(f"         ↑ new best ({val_acc:.3%}) saved to {best_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3%}")
    return model


# ---------------------------------------------------------------------------
# Inference helper (for use in your bot)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> ProtossModel:
    """Load a saved checkpoint for inference inside the bot."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = ProtossModel(
        obs_size=ckpt["obs_size"],
        num_actions=ckpt["num_actions"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_action(model: ProtossModel, obs: list[float], device: str = "cpu") -> int:
    """
    Given a raw observation vector (from ObservationWrapper.get_observation),
    return the predicted action id.
    """
    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    return int(logits.argmax(1).item())


if __name__ == "__main__":
    train()
