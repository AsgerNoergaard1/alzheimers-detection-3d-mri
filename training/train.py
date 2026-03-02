"""
Supervised training for 3D CNN classification of CN vs AD.

Reads train/val CSVs produced by training/splits.py, builds the specified model,
and trains with early stopping based on validation loss. Training curves and the
best checkpoint are saved to RUNS_DIR.

Configuration is controlled by the constants at the top of the file. To switch
models or hyperparameters, edit those constants before running.

Usage:
    python training/train.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn import build_cnn_3d, count_parameters
from training.dataset import ADNIDataset3D
from training.transforms import get_eval_transforms_3d, get_train_transforms_3d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "./data"
RUNS_DIR = "./runs_3d"
os.makedirs(RUNS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_3d.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_3d.csv")

MODEL_NAME   = "monai_densenet121"
BATCH_SIZE   = 2
LR           = 5e-4
EPOCHS       = 500
PATIENCE     = 30
DROPOUT      = 0.2
TARGET_SHAPE = (64, 112, 112)
NUM_WORKERS  = 4
NUM_CLASSES  = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def plot_curves(train_losses, val_losses, train_accs, val_accs, out_file):
    """Save a two-panel loss/accuracy training curve plot."""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_3d_model():
    print(f"\n[SETUP] Device: {DEVICE}")
    print(f"[MODEL] {MODEL_NAME}")

    train_ds = ADNIDataset3D(TRAIN_CSV, transform=get_train_transforms_3d(TARGET_SHAPE))
    val_ds   = ADNIDataset3D(VAL_CSV,   transform=get_eval_transforms_3d(TARGET_SHAPE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Balanced class weights to handle any label imbalance
    df_train = pd.read_csv(TRAIN_CSV)
    y_train = df_train["Group"].map({"CN": 0, "AD": 1}).to_numpy()
    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"[Weights] {class_weights_tensor.cpu().numpy()}")

    model = build_cnn_3d(MODEL_NAME, num_classes=NUM_CLASSES, dropout=DROPOUT).to(DEVICE)
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_train_loss = float("inf")
    best_train_acc = 0.0
    no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}  lr={optimizer.param_groups[0]['lr']:.2e}")

        # --- Training ---
        model.train()
        tr_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc="Train", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = tr_loss / total
        train_acc  = correct / total

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        all_preds, all_labels_val = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss_sum += criterion(outputs, labels).item() * imgs.size(0)
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        val_loss = val_loss_sum / len(val_loader.dataset)

        cm = confusion_matrix(all_labels_val, all_preds)
        tn, fp, fn, tp = cm.ravel()
        val_acc = (tn + tp) / (tn + fp + fn + tp)

        cn_recall   = tn / (tn + fp) if (tn + fp) > 0 else 0
        ad_recall   = tp / (tp + fn) if (tp + fn) > 0 else 0
        cn_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        ad_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        cn_f1 = 2 * cn_precision * cn_recall / (cn_precision + cn_recall) if (cn_precision + cn_recall) > 0 else 0
        ad_f1 = 2 * ad_precision * ad_recall / (ad_precision + ad_recall) if (ad_precision + ad_recall) > 0 else 0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Train  Loss={train_loss:.4f}  Acc={train_acc:.4f} | "
            f"Val  Loss={val_loss:.4f}  Acc={val_acc:.4f}"
        )
        print(f"  CN  recall={cn_recall:.4f}  precision={cn_precision:.4f}  f1={cn_f1:.4f}")
        print(f"  AD  recall={ad_recall:.4f}  precision={ad_precision:.4f}  f1={ad_f1:.4f}")
        print(f"  Confusion matrix: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

        if cn_recall < 0.5 or ad_recall < 0.5:
            print("  WARNING: severely imbalanced predictions this epoch.")

        scheduler.step()

        # --- Early stopping and checkpointing ---
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(RUNS_DIR, "best_model_3d.pt"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(RUNS_DIR, "best_model_val_acc.pt"))
            print(f"  Best val acc checkpoint saved ({best_val_acc:.4f})")

        if train_loss < best_train_loss - 1e-4:
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(RUNS_DIR, "best_model_train_loss.pt"))

        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), os.path.join(RUNS_DIR, "best_model_train_acc.pt"))

    plot_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(RUNS_DIR, "training_curves_3d.png"),
    )

    print("\nTraining completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(RUNS_DIR, 'best_model_3d.pt')}")


if __name__ == "__main__":
    train_3d_model()
