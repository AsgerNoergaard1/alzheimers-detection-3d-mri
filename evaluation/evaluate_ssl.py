"""
Evaluate a fine-tuned SSL encoder on the held-out test set.

Uses the same evaluation logic as evaluation/evaluate.py but loads the
classifier created by ssl/finetune.py (pretrained encoder + classification head).

Usage:
    python evaluation/evaluate_ssl.py
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from models.simclr import create_classifier_from_pretrained
from training.dataset import ADNIDataset3D
from training.transforms import get_eval_transforms_3d


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR     = "./data"
PRETRAIN_DIR = "./simclr_pretrain"
RUNS_DIR     = "./resnet34_finetune_results"
os.makedirs(RUNS_DIR, exist_ok=True)

PRETRAINED_ENCODER_PATH = os.path.join(PRETRAIN_DIR, "pretrained_encoder_best.pt")
TEST_CSV        = os.path.join(DATA_DIR, "test_3d.csv")
BEST_MODEL_PATH = os.path.join(RUNS_DIR, "best_model_pretrained.pt")

BATCH_SIZE   = 1
NUM_WORKERS  = 2
NUM_CLASSES  = 2
TARGET_SHAPE = (60, 112, 112)
CLASS_NAMES  = ["CN", "AD"]
FREEZE_ENCODER = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_ssl():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Test CSV: {TEST_CSV}")

    test_df = pd.read_csv(TEST_CSV)
    test_ds = ADNIDataset3D(TEST_CSV, transform=get_eval_transforms_3d(TARGET_SHAPE))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = create_classifier_from_pretrained(
        pretrained_encoder_path=PRETRAINED_ENCODER_PATH,
        num_classes=NUM_CLASSES,
        spatial_dims=3,
        in_channels=1,
        freeze_encoder=FREEZE_ENCODER,
        encoder_type="resnet",
        resnet_depth=34,
    )
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"[INFO] Loaded weights from: {BEST_MODEL_PATH}")

    all_preds, all_labels, all_probs, all_indices = [], [], [], []

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_indices.extend([idx] * len(labels))

    all_labels  = np.array(all_labels)
    all_preds   = np.array(all_preds)
    all_probs   = np.array(all_probs)
    all_indices = np.array(all_indices)

    # --- Misclassification analysis ---
    misc_mask = all_preds != all_labels
    misc_info = []
    for idx in all_indices[misc_mask]:
        row = test_df.iloc[idx]
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        misc_info.append({
            "index":           idx,
            "subject":         row["Subject"],
            "image_id":        row.get("Image Data ID", "N/A"),
            "path":            row["Path"],
            "true_label":      CLASS_NAMES[true_label],
            "predicted_label": CLASS_NAMES[pred_label],
            "confidence":      all_probs[idx, pred_label],
            "true_class_prob": all_probs[idx, true_label],
            "age":             row.get("Age", "N/A"),
            "sex":             row.get("Sex", "N/A"),
        })

    misc_info.sort(key=lambda x: x["confidence"], reverse=True)
    false_positives = [m for m in misc_info if m["predicted_label"] == "AD"]
    false_negatives = [m for m in misc_info if m["predicted_label"] == "CN"]
    repeat_subjects = {
        s: c for s, c in Counter(m["subject"] for m in misc_info).items() if c > 1
    }

    # --- Classification report ---
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    print("\n" + report)

    report_path = os.path.join(RUNS_DIR, "test_scores_3d.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\nCLASSIFICATION REPORT\n" + "=" * 80 + "\n\n")
        f.write(report + "\n\n")
        f.write("=" * 80 + "\nMISCLASSIFICATION ANALYSIS\n" + "=" * 80 + "\n\n")
        f.write(f"Total: {len(misc_info)} / {len(all_labels)}\n")
        f.write(f"False positives (CN->AD): {len(false_positives)}\n")
        f.write(f"False negatives (AD->CN): {len(false_negatives)}\n\n")
        if repeat_subjects:
            f.write("Subjects with multiple misclassifications:\n")
            for subj, count in sorted(repeat_subjects.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {subj}: {count} scans\n")
            f.write("\n")
        f.write("-" * 80 + "\n")
        for i, m in enumerate(misc_info, 1):
            f.write(
                f"[{i}] Subject: {m['subject']}  True: {m['true_label']}"
                f"  Predicted: {m['predicted_label']} (conf={m['confidence']:.4f})\n"
                f"     True prob: {m['true_class_prob']:.4f}"
                f"  Age: {m['age']}  Sex: {m['sex']}\n"
                f"     {m['path']}\n\n"
            )
    print(f"[SAVED] Report -> {report_path}")

    pd.DataFrame(misc_info).to_csv(
        os.path.join(RUNS_DIR, "misclassified_samples_3d.csv"), index=False
    )

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES).plot(
        cmap="Blues", xticks_rotation=45
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RUNS_DIR, "confusion_matrix_3d.png"))
    plt.close()

    # --- ROC / AUC ---
    if NUM_CLASSES == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], "navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (SSL Fine-tuned)")
        plt.legend(loc="lower right"); plt.tight_layout()
        plt.savefig(os.path.join(RUNS_DIR, "auc_curve_3d.png"))
        plt.close()
        print(f"AUC: {auc:.4f}")
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr")
        print(f"Multi-class AUC (OvR): {auc:.4f}")

    acc = np.mean(all_preds == all_labels)
    print(f"\nAccuracy: {acc:.4f}  |  Test samples: {len(test_ds)}")
    print(f"Misclassified: {len(misc_info)}")

    with open(os.path.join(RUNS_DIR, "summary_metrics_3d.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Total samples: {len(all_labels)}\n")
        f.write(f"Misclassified: {len(misc_info)}\n")
        f.write(f"False positives (CN->AD): {len(false_positives)}\n")
        f.write(f"False negatives (AD->CN): {len(false_negatives)}\n")


if __name__ == "__main__":
    evaluate_ssl()
