"""
Knowledge distillation training: student learns from a pretrained teacher.

The distillation loss is a weighted combination of:
    - KL divergence between softened teacher and student predictions (temperature-scaled)
    - Standard cross-entropy with true labels (class-weighted)

Following Hinton et al. (2015), KL gradients are scaled by T² to maintain
consistent magnitude regardless of temperature.

Configuration is set via the constants below. The teacher checkpoint must
come from a model trained with models/resnet_se_garb.py.

Usage:
    python training/train_distillation.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.resnet_se_garb import build_enhanced_resnet_3d, count_parameters
from models.student import build_student_model_3d
from training.dataset import ADNIDataset3D
from training.transforms import get_eval_transforms_3d, get_train_transforms_3d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "./data"
RUNS_DIR = "./runs_distillation"
os.makedirs(RUNS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_3d.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_3d.csv")

# Teacher
TEACHER_MODEL       = "enhanced_resnet34"
TEACHER_CHECKPOINT  = "resnet_se_garb_model/best_model_3d.pt"
TEACHER_ENHANCEMENT = "garb_layer3"

# Student
STUDENT_MODEL = "tiny_cnn"

# Distillation hyperparameters
TEMPERATURE = 4.0   # Higher temperature → softer teacher distribution
ALPHA       = 0.7   # Weight on the distillation loss; (1 - ALPHA) on hard-label CE

# Training
BATCH_SIZE   = 4
LR           = 5e-4
EPOCHS       = 500
PATIENCE     = 30
DROPOUT      = 0.2
TARGET_SHAPE = (64, 112, 112)
NUM_WORKERS  = 4
NUM_CLASSES  = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------

class DistillationLoss(nn.Module):
    """
    Combined distillation + classification loss.

    total = alpha * KL(student_soft || teacher_soft) * T²
            + (1 - alpha) * CE(student_logits, labels)

    Args:
        temperature: Softening temperature applied to both distributions.
        alpha: Weight on the distillation term.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        student_loss = self.ce_loss(student_logits, labels)
        total = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return total, distillation_loss, student_loss


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def plot_curves(train_losses, val_losses, train_accs, val_accs,
                distill_losses, student_losses, out_file):
    """Save a 2×2 plot showing total loss, accuracy, and loss components."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(train_losses, label="Train Total Loss")
    axes[0, 0].plot(val_losses, label="Val Total Loss")
    axes[0, 0].legend(); axes[0, 0].set_title("Total Loss")

    axes[0, 1].plot(train_accs, label="Train Acc")
    axes[0, 1].plot(val_accs, label="Val Acc")
    axes[0, 1].legend(); axes[0, 1].set_title("Accuracy")

    axes[1, 0].plot(distill_losses, color="orange", label="Distillation Loss (KL)")
    axes[1, 0].legend(); axes[1, 0].set_title("Distillation Loss")

    axes[1, 1].plot(student_losses, color="green", label="Student CE Loss")
    axes[1, 1].legend(); axes[1, 1].set_title("Student Classification Loss")

    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()


def load_teacher(model_name, checkpoint_path, device, enhancement):
    """Load and freeze a pretrained teacher model."""
    print(f"\n[TEACHER] {model_name} from {checkpoint_path}")
    teacher = build_enhanced_resnet_3d(
        model_name,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        input_size=TARGET_SHAPE,
        enhancement=enhancement,
    )
    teacher.load_state_dict(torch.load(checkpoint_path, map_location=device))
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad = False
    total, _ = count_parameters(teacher)
    print(f"[TEACHER] {total:,} parameters (frozen)")
    return teacher


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_with_distillation():
    print(f"\n[SETUP] Device: {DEVICE}")
    print(f"[TEACHER] {TEACHER_MODEL}  enhancement={TEACHER_ENHANCEMENT}")
    print(f"[STUDENT] {STUDENT_MODEL}")
    print(f"[DISTIL]  temperature={TEMPERATURE}  alpha={ALPHA}")

    train_ds = ADNIDataset3D(TRAIN_CSV, transform=get_train_transforms_3d(TARGET_SHAPE))
    val_ds   = ADNIDataset3D(VAL_CSV,   transform=get_eval_transforms_3d(TARGET_SHAPE))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    teacher = load_teacher(TEACHER_MODEL, TEACHER_CHECKPOINT, DEVICE, TEACHER_ENHANCEMENT)

    print(f"\n[STUDENT] Building {STUDENT_MODEL}")
    student = build_student_model_3d(STUDENT_MODEL, num_classes=NUM_CLASSES,
                                     dropout=DROPOUT, input_size=TARGET_SHAPE).to(DEVICE)
    s_total, s_trainable = count_parameters(student)
    t_total, _ = count_parameters(teacher)
    reduction = (1 - s_total / t_total) * 100
    print(f"[STUDENT] {s_total:,} parameters ({reduction:.1f}% reduction vs teacher)")

    df_train = pd.read_csv(TRAIN_CSV)
    y_train = df_train["Group"].map({"CN": 0, "AD": 1}).to_numpy()
    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    ce_weighted = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(DEVICE))

    criterion = DistillationLoss(temperature=TEMPERATURE, alpha=ALPHA)
    optimizer = optim.AdamW(student.parameters(), lr=LR, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)

    best_val_loss = float("inf")
    no_improve = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    distill_losses, student_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}  lr={optimizer.param_groups[0]['lr']:.2e}")

        # --- Training ---
        student.train()
        tr_loss = tr_distill = tr_student = correct = total = 0

        for imgs, labels in tqdm(train_loader, desc="Train", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits = teacher(imgs)

            optimizer.zero_grad()
            student_logits = student(imgs)

            _, distill_loss, _ = criterion(student_logits, teacher_logits, labels)
            stud_loss_w = ce_weighted(student_logits, labels)
            total_loss  = ALPHA * distill_loss + (1 - ALPHA) * stud_loss_w

            total_loss.backward()
            optimizer.step()

            n = imgs.size(0)
            tr_loss    += total_loss.item() * n
            tr_distill += distill_loss.item() * n
            tr_student += stud_loss_w.item() * n
            correct    += (student_logits.argmax(1) == labels).sum().item()
            total      += n

        train_loss   = tr_loss / total
        train_distill = tr_distill / total
        train_student = tr_student / total
        train_acc    = correct / total

        # --- Validation ---
        student.eval()
        vl_loss = vl_distill = vl_student = 0.0
        all_preds, all_labels_val = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Val", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                t_logits = teacher(imgs)
                s_logits = student(imgs)
                _, d_loss, _ = criterion(s_logits, t_logits, labels)
                sw_loss = ce_weighted(s_logits, labels)
                n = imgs.size(0)
                vl_loss    += (ALPHA * d_loss + (1 - ALPHA) * sw_loss).item() * n
                vl_distill += d_loss.item() * n
                vl_student += sw_loss.item() * n
                all_preds.extend(s_logits.argmax(1).cpu().numpy())
                all_labels_val.extend(labels.cpu().numpy())

        val_loss    = vl_loss / len(val_loader.dataset)
        val_distill = vl_distill / len(val_loader.dataset)
        val_student = vl_student / len(val_loader.dataset)

        cm = confusion_matrix(all_labels_val, all_preds)
        tn, fp, fn, tp = cm.ravel()
        val_acc = (tn + tp) / (tn + fp + fn + tp)
        cn_recall   = tn / (tn + fp) if (tn + fp) > 0 else 0
        ad_recall   = tp / (tp + fn) if (tp + fn) > 0 else 0

        train_losses.append(train_loss);  val_losses.append(val_loss)
        train_accs.append(train_acc);     val_accs.append(val_acc)
        distill_losses.append(val_distill)
        student_losses.append(val_student)

        print(
            f"Train  Loss={train_loss:.4f} (distill={train_distill:.4f}, ce={train_student:.4f})"
            f"  Acc={train_acc:.4f}"
        )
        print(
            f"Val    Loss={val_loss:.4f} (distill={val_distill:.4f}, ce={val_student:.4f})"
            f"  Acc={val_acc:.4f}"
        )
        print(f"  CN recall={cn_recall:.4f}  AD recall={ad_recall:.4f}")

        if cn_recall < 0.5 or ad_recall < 0.5:
            print("  WARNING: severely imbalanced predictions this epoch.")

        scheduler.step()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
            save_path = os.path.join(RUNS_DIR, f"best_student_{STUDENT_MODEL}.pt")
            torch.save(student.state_dict(), save_path)
            print(f"  Best student checkpoint saved ({best_val_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print("Early stopping triggered.")
                break

    plot_curves(
        train_losses, val_losses, train_accs, val_accs,
        distill_losses, student_losses,
        os.path.join(RUNS_DIR, f"training_curves_{STUDENT_MODEL}.png"),
    )

    config_path = os.path.join(RUNS_DIR, f"config_{STUDENT_MODEL}.txt")
    with open(config_path, "w") as f:
        f.write(f"Teacher: {TEACHER_MODEL}  enhancement={TEACHER_ENHANCEMENT}\n")
        f.write(f"Teacher checkpoint: {TEACHER_CHECKPOINT}\n")
        f.write(f"Teacher parameters: {t_total:,}\n")
        f.write(f"Student: {STUDENT_MODEL}\n")
        f.write(f"Student parameters: {s_total:,}\n")
        f.write(f"Parameter reduction: {reduction:.1f}%\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write(f"Alpha: {ALPHA}\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n")

    print("\nDistillation training completed.")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_with_distillation()
