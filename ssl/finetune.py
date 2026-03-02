"""
Fine-tune a SimCLR-pretrained encoder for supervised CN/AD classification.

Loads the encoder checkpoint produced by ssl/pretrain.py, attaches a
classification head (already part of the MONAI backbone), and trains on
the supervised data splits. Optionally trains from scratch as a baseline
for comparing the value of pretraining.

Configuration is set via the constants at the top of the file.

Usage:
    python ssl/finetune.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.simclr import create_classifier_from_pretrained
from monai.networks.nets import DenseNet121
from training.dataset import ADNIDataset3D
from training.transforms import get_eval_transforms_3d, get_train_transforms_3d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = "./data"
PRETRAIN_DIR = "./simclr_pretrain"
FINETUNE_DIR = "./finetune_results"
os.makedirs(FINETUNE_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_3d.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_3d.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_3d.csv")

PRETRAINED_ENCODER_PATH = os.path.join(PRETRAIN_DIR, "pretrained_encoder_best.pt")

BATCH_SIZE   = 4
LR           = 3e-4
EPOCHS       = 100
PATIENCE     = 20
TARGET_SHAPE = (60, 112, 112)
NUM_WORKERS  = 4
NUM_CLASSES  = 2

USE_PRETRAINED = True   # Set to False to train the same architecture from scratch
FREEZE_ENCODER = False  # Set to True to only train the classification head

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def plot_curves(train_losses, val_losses, train_accs, val_accs, out_file):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend(); plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.legend(); plt.title("Accuracy")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, out_file):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["CN", "AD"], yticklabels=["CN", "AD"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def evaluate_model(model, dataloader, criterion, device):
    """Run one evaluation pass; return (avg_loss, accuracy, preds, labels)."""
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            total_loss += criterion(outputs, labels).item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ---------------------------------------------------------------------------
# Main fine-tuning routine
# ---------------------------------------------------------------------------

def train_classifier():
    mode = "PRETRAINED" if USE_PRETRAINED else "FROM-SCRATCH"
    print(f"\nFine-tuning AD/CN Classifier — {mode}")
    print(f"Device:  {DEVICE}")
    print(f"Output:  {FINETUNE_DIR}\n")

    train_ds = ADNIDataset3D(TRAIN_CSV, transform=get_train_transforms_3d(TARGET_SHAPE))
    val_ds   = ADNIDataset3D(VAL_CSV,   transform=get_eval_transforms_3d(TARGET_SHAPE))
    test_ds  = ADNIDataset3D(TEST_CSV,  transform=get_eval_transforms_3d(TARGET_SHAPE))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    df_train = pd.read_csv(TRAIN_CSV)
    y_train  = df_train["Group"].map({"CN": 0, "AD": 1}).to_numpy()
    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
    print(f"Class weights: {class_weights_tensor.cpu().numpy()}")

    if USE_PRETRAINED:
        model = create_classifier_from_pretrained(
            pretrained_encoder_path=PRETRAINED_ENCODER_PATH,
            num_classes=NUM_CLASSES,
            spatial_dims=3,
            in_channels=1,
            freeze_encoder=FREEZE_ENCODER,
        )
    else:
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES)
        print("Training from scratch (no pretraining)")

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_train_loss = float("inf")
    best_train_acc = 0.0
    no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    tag = "pretrained" if USE_PRETRAINED else "scratch"

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        # --- Training ---
        model.train()
        tr_loss = correct = total = 0

        for imgs, labels in tqdm(train_loader, desc="Train", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tr_loss  += loss.item() * imgs.size(0)
            correct  += (outputs.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_loss = tr_loss / total
        train_acc  = correct / total

        # --- Validation ---
        val_loss_sum = correct_v = total_v = 0

        model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss_sum += criterion(outputs, labels).item() * imgs.size(0)
                correct_v    += (outputs.argmax(1) == labels).sum().item()
                total_v      += labels.size(0)

        val_loss = val_loss_sum / total_v
        val_acc  = correct_v / total_v

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc);    val_accs.append(val_acc)

        print(f"Train  Loss={train_loss:.4f}  Acc={train_acc:.4f} | Val  Loss={val_loss:.4f}  Acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(FINETUNE_DIR, f"best_model_{tag}.pt"))
            print(f"  Best model saved (val_loss={best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(FINETUNE_DIR, f"best_model_val_acc_{tag}.pt"))
        if train_loss < best_train_loss - 1e-4:
            best_train_loss = train_loss
            torch.save(model.state_dict(), os.path.join(FINETUNE_DIR, f"best_model_train_loss_{tag}.pt"))
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            torch.save(model.state_dict(), os.path.join(FINETUNE_DIR, f"best_model_train_acc_{tag}.pt"))

    plot_curves(
        train_losses, val_losses, train_accs, val_accs,
        os.path.join(FINETUNE_DIR, f"training_curves_{tag}.png"),
    )

    # --- Test set evaluation ---
    model.load_state_dict(torch.load(os.path.join(FINETUNE_DIR, f"best_model_{tag}.pt")))
    test_loss, test_acc, test_preds, test_labels = evaluate_model(model, test_loader, criterion, DEVICE)

    report = classification_report(test_labels, test_preds, target_names=["CN", "AD"], digits=4)
    print(f"\n{report}")

    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average="binary")

    plot_confusion_matrix(test_labels, test_preds, os.path.join(FINETUNE_DIR, f"confusion_matrix_{tag}.png"))

    with open(os.path.join(FINETUNE_DIR, "test_scores_3d.txt"), "w") as f:
        f.write(report)

    with open(os.path.join(FINETUNE_DIR, f"results_{tag}.txt"), "w") as f:
        f.write(f"AD/CN Classification — {mode}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best val loss:   {best_val_loss:.4f}\n")
        f.write(f"Test loss:       {test_loss:.4f}\n")
        f.write(f"Test accuracy:   {test_acc:.4f}\n")
        f.write(f"Test precision:  {precision:.4f}\n")
        f.write(f"Test recall:     {recall:.4f}\n")
        f.write(f"Test F1:         {f1:.4f}\n")

    print(f"\nTest accuracy: {test_acc:.4f}  |  F1: {f1:.4f}")
    print(f"Results saved to {FINETUNE_DIR}")
    return test_acc, test_loss


if __name__ == "__main__":
    train_classifier()
