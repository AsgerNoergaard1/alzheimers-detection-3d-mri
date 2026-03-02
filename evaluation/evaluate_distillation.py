"""
Evaluate and compare teacher, distilled student, and (optionally) a baseline
student trained from scratch.

Produces:
    - ROC curve comparison
    - Confusion matrix comparison
    - Metric bar chart
    - Efficiency table (params, inference time, metrics)
    - Detailed text report

Usage:
    python evaluation/evaluate_distillation.py
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from models.resnet_se_garb import build_enhanced_resnet_3d, count_parameters
from models.student import build_student_model_3d
from training.dataset import ADNIDataset3D
from training.transforms import get_eval_transforms_3d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = "./data"
RUNS_DIR     = "./runs_distillation"
COMPARE_DIR  = os.path.join(RUNS_DIR, "comparison")
os.makedirs(COMPARE_DIR, exist_ok=True)

TEST_CSV = os.path.join(DATA_DIR, "test_3d.csv")

TEACHER_MODEL      = "enhanced_resnet34"
TEACHER_CHECKPOINT = "resnet_se_garb_model/best_model_3d.pt"

STUDENT_MODEL      = "tiny_cnn"
STUDENT_CHECKPOINT = os.path.join(RUNS_DIR, f"best_student_{STUDENT_MODEL}.pt")

# Optional: path to a student trained from scratch (no distillation) for comparison.
BASELINE_CHECKPOINT = "tiny_cnn_evaluation/best_model_3d.pt"

BATCH_SIZE   = 1
NUM_WORKERS  = 2
NUM_CLASSES  = 2
TARGET_SHAPE = (64, 112, 112)
CLASS_NAMES  = ["CN", "AD"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate_model(model, loader, model_name: str) -> dict:
    """Run evaluation and return a results dictionary."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    inference_times = []

    print(f"\n[{model_name}] Evaluating...")

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            t0 = time.time()
            outputs = model(imgs)
            inference_times.append(time.time() - t0)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(np.argmax(probs, axis=1))
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1
    )
    auc = roc_auc_score(all_labels, all_probs[:, 1])

    return {
        "model_name":          model_name,
        "accuracy":            (tn + tp) / (tn + fp + fn + tp),
        "precision":           precision,
        "recall":              recall,
        "f1":                  f1,
        "auc":                 auc,
        "cn_recall":           tn / (tn + fp) if (tn + fp) > 0 else 0,
        "ad_recall":           tp / (tp + fn) if (tp + fn) > 0 else 0,
        "cn_precision":        tn / (tn + fn) if (tn + fn) > 0 else 0,
        "ad_precision":        tp / (tp + fp) if (tp + fp) > 0 else 0,
        "confusion_matrix":    cm,
        "predictions":         all_preds,
        "labels":              all_labels,
        "probabilities":       all_probs,
        "avg_inference_ms":    np.mean(inference_times) * 1000,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc_comparison(results_dict, save_path):
    colors = ["blue", "green", "orange", "red"]
    plt.figure(figsize=(8, 6))
    for idx, (name, res) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(res["labels"], res["probabilities"][:, 1])
        plt.plot(fpr, tpr, color=colors[idx], lw=2, label=f"{name} (AUC={res['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison"); plt.legend(loc="lower right")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] ROC comparison -> {save_path}")


def plot_confusion_matrices(results_dict, save_path):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for idx, (name, res) in enumerate(results_dict.items()):
        ConfusionMatrixDisplay(confusion_matrix=res["confusion_matrix"],
                               display_labels=CLASS_NAMES).plot(
            ax=axes[idx], cmap="Blues", xticks_rotation=45
        )
        axes[idx].set_title(f"{name}\nAcc: {res['accuracy']:.3f}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Confusion matrices -> {save_path}")


def plot_metric_bars(results_dict, save_path):
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    n_models = len(results_dict)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x     = np.arange(len(metrics))
    width = 0.8 / n_models

    for idx, (name, res) in enumerate(results_dict.items()):
        values = [res[m] for m in metrics]
        offset = width * (idx - n_models / 2 + 0.5)
        bars = ax.bar(x + offset, values, width, label=name, color=colors[idx], alpha=0.8)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.05]); ax.legend(loc="lower right")
    ax.set_title("Performance Comparison"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Metric comparison -> {save_path}")


def save_report(results_dict, save_path):
    with open(save_path, "w") as f:
        f.write("=" * 80 + "\nKNOWLEDGE DISTILLATION EVALUATION REPORT\n" + "=" * 80 + "\n")
        for name, res in results_dict.items():
            f.write(f"\n{'='*80}\nMODEL: {name}\n{'='*80}\n\n")
            f.write(f"Accuracy:        {res['accuracy']:.4f}\n")
            f.write(f"Precision:       {res['precision']:.4f}\n")
            f.write(f"Recall:          {res['recall']:.4f}\n")
            f.write(f"F1-Score:        {res['f1']:.4f}\n")
            f.write(f"AUC:             {res['auc']:.4f}\n")
            f.write(f"Inference time:  {res['avg_inference_ms']:.2f} ms\n\n")
            f.write(f"CN recall:       {res['cn_recall']:.4f}  |  precision: {res['cn_precision']:.4f}\n")
            f.write(f"AD recall:       {res['ad_recall']:.4f}  |  precision: {res['ad_precision']:.4f}\n\n")
            tn, fp, fn, tp = res["confusion_matrix"].ravel()
            f.write(f"Confusion matrix:\n")
            f.write(f"                  Pred CN    Pred AD\n")
            f.write(f"  Actual CN       {tn:6d}    {fp:6d}\n")
            f.write(f"  Actual AD       {fn:6d}    {tp:6d}\n\n")

        if len(results_dict) >= 2:
            keys = list(results_dict.keys())
            teacher = results_dict[keys[0]]
            student = results_dict[keys[1]]
            f.write("=" * 80 + "\nSTUDENT vs TEACHER\n" + "=" * 80 + "\n\n")
            f.write(f"Accuracy retention: {student['accuracy'] / teacher['accuracy'] * 100:.1f}%\n")
            f.write(f"AUC retention:      {student['auc'] / teacher['auc'] * 100:.1f}%\n")
            f.write(f"Speed improvement:  {teacher['avg_inference_ms'] / student['avg_inference_ms']:.2f}x\n")
    print(f"[SAVED] Detailed report -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_distillation():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Test CSV: {TEST_CSV}")

    test_ds = ADNIDataset3D(TEST_CSV, transform=get_eval_transforms_3d(TARGET_SHAPE))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"[INFO] Test samples: {len(test_ds)}")

    results_dict = {}

    # --- Teacher ---
    teacher = build_enhanced_resnet_3d(
        TEACHER_MODEL, num_classes=NUM_CLASSES,
        input_size=TARGET_SHAPE, enhancement="garb_layer3",
    )
    teacher.load_state_dict(torch.load(TEACHER_CHECKPOINT, map_location=DEVICE))
    teacher.to(DEVICE)
    teacher_params, _ = count_parameters(teacher)
    print(f"[Teacher] {teacher_params:,} parameters")
    results_dict["Teacher"] = evaluate_model(teacher, test_loader, "Teacher")

    # --- Student ---
    student = build_student_model_3d(STUDENT_MODEL, num_classes=NUM_CLASSES, input_size=TARGET_SHAPE)
    if not os.path.exists(STUDENT_CHECKPOINT):
        print(f"[ERROR] Student checkpoint not found: {STUDENT_CHECKPOINT}")
        print("[ERROR] Train with training/train_distillation.py first.")
        return
    student.load_state_dict(torch.load(STUDENT_CHECKPOINT, map_location=DEVICE))
    student.to(DEVICE)
    student_params, _ = count_parameters(student)
    reduction = (1 - student_params / teacher_params) * 100
    print(f"[Student] {student_params:,} parameters ({reduction:.1f}% reduction)")
    results_dict["Student (Distilled)"] = evaluate_model(student, test_loader, "Student (Distilled)")

    # --- Baseline (optional) ---
    if BASELINE_CHECKPOINT and os.path.exists(BASELINE_CHECKPOINT):
        baseline = build_student_model_3d(STUDENT_MODEL, num_classes=NUM_CLASSES, input_size=TARGET_SHAPE)
        baseline.load_state_dict(torch.load(BASELINE_CHECKPOINT, map_location=DEVICE))
        baseline.to(DEVICE)
        results_dict["Baseline (No Distill)"] = evaluate_model(baseline, test_loader, "Baseline (No Distill)")

    # --- Plots and report ---
    plot_roc_comparison(results_dict,    os.path.join(COMPARE_DIR, "roc_comparison.png"))
    plot_confusion_matrices(results_dict, os.path.join(COMPARE_DIR, "confusion_matrices.png"))
    plot_metric_bars(results_dict,       os.path.join(COMPARE_DIR, "metric_comparison.png"))
    save_report(results_dict,            os.path.join(COMPARE_DIR, "detailed_report.txt"))

    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    for name, res in results_dict.items():
        print(
            f"{name:<25} | Acc: {res['accuracy']:.4f} | AUC: {res['auc']:.4f}"
            f" | F1: {res['f1']:.4f} | Time: {res['avg_inference_ms']:.2f} ms"
        )
    print(f"\nResults saved to: {COMPARE_DIR}")


if __name__ == "__main__":
    evaluate_distillation()
