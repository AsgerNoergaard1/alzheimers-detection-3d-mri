"""
Generate 3D Grad-CAM visualizations for Alzheimer's detection.

Processes the full test set, computes Grad-CAM for each sample, selects the
top-N most confident predictions per category (CN correct, CN incorrect,
AD correct, AD incorrect), and saves overlaid axial-slice visualizations.

Usage:
    python xai/generate_visualizations.py
"""

import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn import build_cnn_3d
from models.resnet_se_garb import build_enhanced_resnet_3d
from training.dataset import ADNIDataset3D
from training.transforms import get_eval_transforms_3d
from xai.grad_cam import GradCAM3D
from xai.targets import ClassifierOutputTarget


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "./data"
TEST_CSV = os.path.join(DATA_DIR, "test_3d.csv")

# Switch between model architectures by changing these two values.
MODEL_NAME       = "monai_densenet121"
BEST_MODEL_PATH  = "densenet_model/best_model_3d.pt"

OUTPUT_DIR   = f"./gradcam_results_{MODEL_NAME}"
TARGET_SHAPE = (64, 112, 112)
BATCH_SIZE   = 1
NUM_CLASSES  = 2
CLASS_NAMES  = ["CN", "AD"]
TOP_N_CONFIDENT = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
for _cat in ["CN_correct", "CN_incorrect", "AD_correct", "AD_incorrect"]:
    os.makedirs(os.path.join(OUTPUT_DIR, _cat), exist_ok=True)


# ---------------------------------------------------------------------------
# NIfTI utilities
# ---------------------------------------------------------------------------

def load_raw_nifti_slice(
    nifti_path: str, slice_idx: int, target_size: tuple = (112, 112)
) -> np.ndarray:
    """Load a single axial slice from a NIfTI file and normalise to [0, 1].

    Args:
        nifti_path: Path to the .nii.gz file.
        slice_idx: Axial slice index in cropped coordinates (0-111).
        target_size: (H, W) to resize to.

    Returns:
        2D array normalised to [0, 1].
    """
    img = nib.load(nifti_path).get_fdata()

    # CenterSpatialCrop removes (193 - 112) // 2 = 40 slices from each end,
    # so cropped index 0 = raw index 40.
    crop_offset = (img.shape[2] - 112) // 2
    raw_idx = crop_offset + slice_idx

    slice_2d = img[:, :, raw_idx]
    slice_2d = slice_2d - slice_2d.min()
    slice_2d = slice_2d / (slice_2d.max() + 1e-8)

    if slice_2d.shape != target_size:
        factors = [target_size[i] / slice_2d.shape[i] for i in range(2)]
        slice_2d = zoom(slice_2d, factors, order=1)

    return slice_2d


def overlay_cam_on_slice(
    mri_slice: np.ndarray, cam_slice: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto a greyscale MRI slice.

    Args:
        mri_slice: 2D array normalised to [0, 1].
        cam_slice: 2D array normalised to [0, 1].
        alpha: Heatmap opacity (0 = transparent, 1 = opaque).

    Returns:
        RGB array with the heatmap overlay.
    """
    if mri_slice.shape != cam_slice.shape:
        factors = [cam_slice.shape[i] / mri_slice.shape[i] for i in range(2)]
        mri_slice = zoom(mri_slice, factors, order=1)

    mri_rgb  = np.stack([mri_slice] * 3, axis=-1)
    heatmap  = plt.get_cmap("jet")(cam_slice)[:, :, :3]
    overlay  = alpha * heatmap + (1 - alpha) * mri_rgb
    overlay  = overlay / (overlay.max() + 1e-8)
    return overlay


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def create_visualization(
    mri_path: str,
    cam_3d: np.ndarray,
    subject_id: str,
    prediction: int,
    confidence: float,
    true_label: int,
) -> plt.Figure:
    """Create a 1×2 figure showing the two axial slices with highest attention.

    The CAM volume has shape (D=64, H=112, W=112) in (sagittal, coronal, axial)
    order.  Transposing to axial-first gives (112, 64, 112), and the two slices
    with the largest summed attention are selected.

    Args:
        mri_path: Path to the original NIfTI file.
        cam_3d: Grad-CAM volume (D, H, W).
        subject_id: Subject identifier string.
        prediction: Predicted class index.
        confidence: Softmax confidence for the predicted class.
        true_label: Ground-truth class index.

    Returns:
        Matplotlib Figure object.
    """
    # Transpose so axial dimension is first: (112_axial, 64, 112)
    cam_axial   = np.transpose(cam_3d, (2, 0, 1))
    slice_sum   = cam_axial.sum(axis=(1, 2))
    top_slices  = np.argsort(slice_sum)[-2:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for col, slice_idx in enumerate(top_slices):
        mri_slice = load_raw_nifti_slice(mri_path, slice_idx)
        cam_slice = cam_axial[slice_idx]
        overlay   = overlay_cam_on_slice(mri_slice, cam_slice, alpha=0.4)

        raw_idx   = 40 + slice_idx
        axes[col].imshow(overlay)
        axes[col].set_title(
            f"Rank #{col+1}: slice {slice_idx} (raw {raw_idx})\n"
            f"Attention sum: {slice_sum[slice_idx]:.0f}",
            fontsize=12,
        )
        axes[col].axis("off")

    pred_class = CLASS_NAMES[prediction]
    true_class = CLASS_NAMES[true_label]
    fig.suptitle(
        f"Subject: {subject_id}\n"
        f"Prediction: {pred_class} (conf: {confidence:.3f})  |  Ground truth: {true_class}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Target-layer selection
# ---------------------------------------------------------------------------

def get_target_layer(model, model_name: str) -> list:
    """Return the Grad-CAM target layer for the given architecture.

    Args:
        model: Loaded model.
        model_name: Architecture name string.

    Returns:
        Single-element list containing the target nn.Module.

    Raises:
        NotImplementedError: For ViT models (requires special handling).
        ValueError: For unknown architectures.
    """
    name = model_name.lower()
    if "densenet" in name:
        return [model.densenet.features.denseblock4]
    if "resnet" in name:
        return [model.resnet.layer4[-1]]
    if "vit" in name:
        raise NotImplementedError("Grad-CAM for ViT requires a reshape transform.")
    raise ValueError(f"Unknown model architecture: {model_name}")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_test_set(model, test_df: pd.DataFrame, test_loader: DataLoader) -> tuple:
    """Run Grad-CAM on the full test set.

    Returns:
        results: List of per-sample result dicts.
        slice_stats: Dict of attention statistics grouped by category.
    """
    target_layers = get_target_layer(model, MODEL_NAME)
    cam = GradCAM3D(model=model, target_layers=target_layers)

    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Target layer: {target_layers[0]}")
    print(f"[INFO] Processing {len(test_df)} test samples...")

    results = []
    slice_stats = {
        "all": [],
        "CN_correct": [], "CN_incorrect": [],
        "AD_correct": [], "AD_incorrect": [],
    }

    for idx, (img, label) in enumerate(tqdm(test_loader, desc="Grad-CAM")):
        img = img.to(DEVICE)
        label_item = label.item()

        with torch.no_grad():
            output = model(img)
            probs  = F.softmax(output, dim=1)
            pred   = torch.argmax(probs, dim=1).item()
            conf   = probs[0, pred].item()

        subject_id  = test_df.iloc[idx]["Subject"]
        nifti_path  = test_df.iloc[idx]["Path"]

        img.requires_grad = True
        grayscale_cam = cam(input_tensor=img, targets=[ClassifierOutputTarget(pred)])
        cam_3d = grayscale_cam[0]  # (D, H, W)

        cam_axial        = np.transpose(cam_3d, (2, 0, 1))
        slice_attention  = cam_axial.sum(axis=(1, 2))
        top_slice        = int(np.argmax(slice_attention))
        top_2_slices     = np.argsort(slice_attention)[-2:][::-1].tolist()

        is_correct = pred == label_item
        category   = ("CN" if label_item == 0 else "AD") + ("_correct" if is_correct else "_incorrect")

        entry = {
            "subject": subject_id,
            "category": category,
            "top_slice": top_slice,
            "top_2_slices": top_2_slices,
            "slice_attention": slice_attention.tolist(),
            "prediction": pred,
            "confidence": conf,
        }
        slice_stats["all"].append(entry)
        slice_stats[category].append(entry)

        results.append({
            "subject_id":    subject_id,
            "nifti_path":    nifti_path,
            "true_label":    label_item,
            "prediction":    pred,
            "confidence":    conf,
            "cam_3d":        cam_3d,
            "is_correct":    is_correct,
            "top_slice":     top_slice,
            "slice_attention": slice_attention,
        })

    return results, slice_stats


def select_top_confident(results: list) -> dict:
    """Group results by category and return the top-N most confident per group."""
    categories = {k: [] for k in ["CN_correct", "CN_incorrect", "AD_correct", "AD_incorrect"]}

    for res in results:
        cat = ("CN" if res["true_label"] == 0 else "AD") + ("_correct" if res["is_correct"] else "_incorrect")
        categories[cat].append(res)

    selected = {}
    for cat, items in categories.items():
        items.sort(key=lambda x: x["confidence"], reverse=True)
        selected[cat] = items[:TOP_N_CONFIDENT]
        print(f"[INFO] {cat}: {len(items)} total, selected top {len(selected[cat])}")

    return selected


def generate_visualizations(selected_samples: dict):
    """Save overlay figures for all selected samples."""
    print("\n[INFO] Generating visualizations...")
    for category, samples in selected_samples.items():
        print(f"  Category: {category}")
        for sample in samples:
            fig = create_visualization(
                sample["nifti_path"],
                sample["cam_3d"],
                sample["subject_id"],
                sample["prediction"],
                sample["confidence"],
                sample["true_label"],
            )
            filename = (
                f"{sample['subject_id']}_pred_{CLASS_NAMES[sample['prediction']]}"
                f"_conf_{sample['confidence']:.3f}.png"
            )
            save_path = os.path.join(OUTPUT_DIR, category, filename)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: {save_path}")


# ---------------------------------------------------------------------------
# Statistics report
# ---------------------------------------------------------------------------

def save_slice_statistics(slice_stats: dict, results: list):
    """Write a text report of slice-level attention statistics."""
    stats_path = os.path.join(OUTPUT_DIR, "slice_attention_statistics.txt")

    with open(stats_path, "w") as f:
        f.write("=" * 80 + "\nSLICE ATTENTION STATISTICS - 3D GRAD-CAM\n" + "=" * 80 + "\n\n")
        f.write(f"Total test samples : {len(results)}\n")
        f.write(f"Model              : {MODEL_NAME}\n")
        f.write(f"Depth              : {TARGET_SHAPE[0]} slices\n\n")

        # Overall
        all_top = [e["top_slice"] for e in slice_stats["all"]]
        all_attn = np.array([e["slice_attention"] for e in slice_stats["all"]])
        mean_attn = all_attn.mean(axis=0)
        top5 = np.argsort(mean_attn)[-5:][::-1]

        f.write("=" * 80 + "\nOVERALL (all test samples)\n" + "=" * 80 + "\n\n")
        f.write("Top 5 slices by mean attention:\n")
        for rank, si in enumerate(top5, 1):
            f.write(f"  {rank}. slice {si}: mean={mean_attn[si]:.2f}, std={all_attn[:, si].std():.2f}\n")

        counts = np.bincount(all_top)
        f.write(f"\nMost common top-attention slice: {counts.argmax()} ({counts.max()}/{len(all_top)} samples)\n")

        # Per category
        for cat in ["CN_correct", "CN_incorrect", "AD_correct", "AD_incorrect"]:
            data = slice_stats[cat]
            f.write(f"\n{'='*80}\n{cat.upper()}\n{'='*80}\n\n")
            if not data:
                f.write("  No samples.\n")
                continue
            cat_top = [e["top_slice"] for e in data]
            cat_attn = np.array([e["slice_attention"] for e in data])
            cat_mean = cat_attn.mean(axis=0)
            cat5 = np.argsort(cat_mean)[-5:][::-1]
            f.write(f"Samples: {len(data)}\n")
            f.write("Top 5 slices by mean attention:\n")
            for rank, si in enumerate(cat5, 1):
                f.write(f"  {rank}. slice {si}: mean={cat_mean[si]:.2f}\n")
            cc = np.bincount(cat_top)
            f.write(f"Most common: slice {cc.argmax()} ({cc.max()}/{len(cat_top)} samples)\n")

        # CN vs AD comparison
        cn_top = [e["top_slice"] for cat in ["CN_correct", "CN_incorrect"] for e in slice_stats[cat]]
        ad_top = [e["top_slice"] for cat in ["AD_correct", "AD_incorrect"] for e in slice_stats[cat]]
        f.write(f"\n{'='*80}\nCN vs AD ATTENTION PATTERNS\n{'='*80}\n\n")
        if cn_top:
            f.write(f"CN mean top slice: {np.mean(cn_top):.1f} (std {np.std(cn_top):.1f})\n")
        if ad_top:
            f.write(f"AD mean top slice: {np.mean(ad_top):.1f} (std {np.std(ad_top):.1f})\n")

    print(f"[SAVED] Slice statistics -> {stats_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("3D GRAD-CAM VISUALIZATION")
    print("=" * 80)
    print(f"Device    : {DEVICE}")
    print(f"Model     : {MODEL_NAME}")
    print(f"Test CSV  : {TEST_CSV}")
    print(f"Output    : {OUTPUT_DIR}")
    print("=" * 80)

    model = build_cnn_3d(MODEL_NAME, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE).eval()

    test_df     = pd.read_csv(TEST_CSV)
    test_ds     = ADNIDataset3D(TEST_CSV, transform=get_eval_transforms_3d(TARGET_SHAPE))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    results, slice_stats = process_test_set(model, test_df, test_loader)
    save_slice_statistics(slice_stats, results)

    selected = select_top_confident(results)
    generate_visualizations(selected)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
