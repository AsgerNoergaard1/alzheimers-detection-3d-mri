"""
SimCLR contrastive pretraining on 3D brain MRI.

Labels are ignored during pretraining — the model learns by distinguishing
augmented views of the same volume from views of different volumes.

This script requires an augmentations module (not included) that provides:
    ContrastiveAugmentation  - creates two augmented views from each input
    get_base_preprocessing   - loads and normalises NIfTI volumes

After pretraining, the encoder is saved to PRETRAIN_DIR and can be loaded
by ssl/finetune.py for supervised fine-tuning.

Configuration is set via the constants at the top of the file.

Usage:
    python ssl/pretrain.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.optim as optim
from monai.data import Dataset
from monai.transforms import Compose
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import ContrastiveAugmentation, get_base_preprocessing  # noqa: F401
from models.simclr import build_simclr_model
from ssl.losses import get_loss_function


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = "./data"
PRETRAIN_DIR = "./simclr_pretrain"
os.makedirs(PRETRAIN_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_3d.csv")

# Model
ENCODER_OUT_DIM = 1024
PROJECTION_DIM  = 128
HIDDEN_DIM      = 512

# Training
EPOCHS            = 200
PATIENCE          = 30
BATCH_SIZE        = 8
ACCUMULATION_STEPS = 4   # Effective batch size = BATCH_SIZE * ACCUMULATION_STEPS
LR               = 1e-4
WEIGHT_DECAY     = 1e-6
USE_AMP          = True

# Loss
LOSS_TYPE   = "ntxent"
TEMPERATURE = 0.5

# Augmentation
TARGET_SHAPE     = (64, 112, 112)
ROTATION_RANGE   = 0.26   # ~15 degrees in radians
SCALE_RANGE      = 0.1
INTENSITY_SCALE  = 0.3
INTENSITY_SHIFT  = 0.1
CONTRAST_GAMMA   = (0.7, 1.5)
NOISE_STD        = 0.1
SMOOTH_SIGMA     = (0.5, 1.0)

# Data loading
NUM_WORKERS = 8
PIN_MEMORY  = True

# Scheduler
SCHEDULER_TYPE = "cosine"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def plot_training_losses(losses_history: list, out_file: str):
    """Save a simple loss-vs-epoch plot."""
    epochs = [entry["epoch"] for entry in losses_history]
    losses = [entry["loss"]  for entry in losses_history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, linewidth=2, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Contrastive Loss")
    plt.title("SimCLR Pretraining Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Loss plot saved: {out_file}")


# ---------------------------------------------------------------------------
# Main pretraining routine
# ---------------------------------------------------------------------------

def train_simclr():
    print(f"\nSimCLR Contrastive Pretraining")
    print(f"Device:        {DEVICE}")
    print(f"Training CSV:  {TRAIN_CSV}")
    print(f"Output dir:    {PRETRAIN_DIR}")
    print(f"Batch size:    {BATCH_SIZE} (effective: {BATCH_SIZE * ACCUMULATION_STEPS})")
    print(f"Target shape:  {TARGET_SHAPE}")
    print(f"Mixed prec:    {USE_AMP}\n")

    train_df = pd.read_csv(TRAIN_CSV)
    print(f"Training samples: {len(train_df)}")

    base_transforms = get_base_preprocessing(keys=["image"], target_shape=TARGET_SHAPE)
    contrastive_aug = ContrastiveAugmentation(
        keys=["image"],
        rotation_range=ROTATION_RANGE,
        scale_range=SCALE_RANGE,
        intensity_scale=INTENSITY_SCALE,
        intensity_shift=INTENSITY_SHIFT,
        contrast_gamma=CONTRAST_GAMMA,
        noise_std=NOISE_STD,
        smooth_sigma=SMOOTH_SIGMA,
    )
    all_transforms = Compose([base_transforms, contrastive_aug.get_transforms()])

    # Labels are deliberately excluded — SSL uses image identity as supervision
    train_files = [{"image": row["Path"]} for _, row in train_df.iterrows()]
    train_dataset = Dataset(data=train_files, transform=all_transforms)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    print(f"Batches per epoch: {len(train_loader)}\n")

    config = {
        "spatial_dims":    3,
        "in_channels":     1,
        "encoder_out_dim": ENCODER_OUT_DIM,
        "projection_dim":  PROJECTION_DIM,
        "hidden_dim":      HIDDEN_DIM,
        "encoder_type":    "resnet",
        "resnet_depth":    50,
    }
    model = build_simclr_model(config).to(DEVICE)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total:,}\n")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if SCHEDULER_TYPE == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    elif SCHEDULER_TYPE == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        scheduler = None

    loss_fn = get_loss_function(loss_type=LOSS_TYPE, temperature=TEMPERATURE)
    scaler  = torch.cuda.amp.GradScaler() if USE_AMP else None

    losses_history = []
    best_loss  = float("inf")
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
        for batch_idx, batch in enumerate(progress):
            view1 = batch["view1"].to(DEVICE)
            view2 = batch["view2"].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                _, proj1 = model(view1)
                _, proj2 = model(view2)
                loss = loss_fn(proj1, proj2) / ACCUMULATION_STEPS

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * ACCUMULATION_STEPS * view1.size(0)
            progress.set_postfix({"loss": f"{loss.item() * ACCUMULATION_STEPS:.4f}"})

        avg_loss = epoch_loss / len(train_dataset)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        losses_history.append({"epoch": epoch, "loss": avg_loss, "lr": current_lr})
        print(f"\nEpoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  lr={current_lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            model.save_encoder(os.path.join(PRETRAIN_DIR, "pretrained_encoder_best.pt"))
            torch.save(model.state_dict(), os.path.join(PRETRAIN_DIR, "simclr_model_best.pt"))
            print("  New best — encoder checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping after {epoch} epochs (no improvement for {PATIENCE} epochs).")
                break

        print("-" * 60)

    plot_training_losses(losses_history, os.path.join(PRETRAIN_DIR, "training_losses.png"))

    print(f"\nPretraining complete.  Best loss: {best_loss:.4f}")
    print(f"Artifacts saved to: {PRETRAIN_DIR}")


if __name__ == "__main__":
    train_simclr()
