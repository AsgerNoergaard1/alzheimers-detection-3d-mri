"""
Preprocessing pipeline for 3D structural MRI volumes.

Each series passes through four stages:
    1. Skull stripping (HD-BET)
    2. N4 bias field correction (ANTs)
    3. Rigid registration to MNI152 space (ANTs)
    4. Intensity normalisation within brain mask (percentile clipping to [0, 1])

An optional fifth stage exports axial PNG slices for quick QC.

Input:  CSV manifest with columns patient, series_uid, nii (path to raw NIfTI).
Output: Preprocessed volumes under <out>/rigid/, per-batch log CSV.

Usage:
    python preprocessing/preprocess.py \\
        --manifest nifti_out/manifest_.csv \\
        --mni templates/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii \\
        --out preproc_out \\
        --start 0 --end 100 --device cuda:0 --skip-done
"""

import atexit
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import warnings
from pathlib import Path

import ants
import imageio.v2 as iio
import nibabel as nib
import numpy as np
import pandas as pd
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def safe_name(s: str) -> str:
    """Sanitise a string for use as a filename component."""
    if s is None:
        return "NA"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def norm_path(p: str) -> Path:
    """Normalise a path string (handle Windows backslashes)."""
    return Path(p.replace("\\", "/"))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Skull stripping (HD-BET)
# ---------------------------------------------------------------------------

def _looks_binary(arr: np.ndarray) -> bool:
    u = np.unique(arr)
    return (len(u) <= 4) and (u.min() >= 0) and (u.max() <= 2)


def run_hdbet(in_path: Path, out_brain: Path, out_mask: Path, device: str = "cpu"):
    """Run HD-BET skull stripping and write brain and mask volumes.

    Handles the case where HD-BET swaps the brain/mask outputs by checking
    whether each file contains binary or continuous values.
    """
    dev_opt = "cuda" if str(device).lower().startswith("cuda") else "cpu"
    out_brain.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    if not str(out_brain).endswith(".nii.gz"):
        out_brain = out_brain.with_suffix(".nii").with_suffix(".nii.gz")
    if not str(out_mask).endswith(".nii.gz"):
        out_mask = out_mask.with_suffix(".nii").with_suffix(".nii.gz")

    tmp = out_brain.parent / f"_hdbet_tmp_{uuid.uuid4().hex}"
    tmp.mkdir(parents=True, exist_ok=True)
    tmp_in = tmp / "case_000.nii.gz"
    shutil.copyfile(str(in_path), str(tmp_in))

    def _run(argv):
        try:
            from hd_bet.run import main as hdbet_main
            old = sys.argv[:]
            sys.argv = argv
            try:
                hdbet_main()
            finally:
                sys.argv = old
        except Exception:
            exe = shutil.which("hd-bet")
            if exe:
                subprocess.run([exe] + argv[1:], check=True)
            else:
                subprocess.run([sys.executable, "-m", "hd_bet.run"] + argv[1:], check=True)

    _run(["hd-bet", "-i", str(tmp), "-o", str(tmp), "-device", dev_opt])
    try:
        _run([
            "hd-bet", "-i", str(tmp), "-o", str(tmp),
            "--save_bet_mask", "--no_bet_image", "-device", dev_opt,
        ])
    except Exception:
        pass

    bet  = next((p for p in tmp.glob("*_bet.nii.gz")), None)
    mask = next((p for p in tmp.glob("*_bet_mask.nii.gz")), None)
    if bet is None:
        bet = tmp / "case_000.nii.gz"
    if mask is None:
        arr = nib.load(str(bet)).get_fdata()
        m   = (arr > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(m, nib.load(str(bet)).affine), str(tmp / "auto_mask.nii.gz"))
        mask = tmp / "auto_mask.nii.gz"

    arr_b = nib.load(str(bet)).get_fdata()
    arr_m = nib.load(str(mask)).get_fdata()

    if _looks_binary(arr_b) and not _looks_binary(arr_m):
        bet, mask = mask, bet
        arr_b, arr_m = arr_m, arr_b

    if _looks_binary(arr_b):
        orig  = nib.load(str(in_path))
        oarr  = orig.get_fdata()
        m     = (arr_m > 0.5).astype(np.float32)
        fixed = (oarr * m).astype(np.float32)
        nib.save(nib.Nifti1Image(fixed, orig.affine, orig.header), str(out_brain))
    else:
        shutil.move(str(bet), str(out_brain))

    m_bin = (arr_m > 0.5).astype(np.uint8)
    nib.save(nib.Nifti1Image(m_bin, nib.load(str(mask)).affine), str(out_mask))
    shutil.rmtree(tmp, ignore_errors=True)


def is_binary_nii(p: Path) -> bool:
    a = nib.load(str(p)).get_fdata()
    u = np.unique(a)
    return (len(u) <= 4) and (u.min() >= 0) and (u.max() <= 2)


# ---------------------------------------------------------------------------
# ANTs steps
# ---------------------------------------------------------------------------

def n4_bias_correct(in_img: Path, in_mask: Path):
    """Apply N4 bias field correction in-place (overwrites in_img)."""
    img = ants.image_read(str(in_img))
    msk = ants.image_read(str(in_mask))
    n4  = ants.n4_bias_field_correction(img, msk)
    ants.image_write(n4, str(in_img))


def rigid_to_mni(mni_path: Path, in_path: Path) -> dict:
    """Compute a rigid registration from in_path to the MNI template."""
    fixed  = ants.image_read(str(mni_path))
    moving = ants.image_read(str(in_path))
    return ants.registration(fixed=fixed, moving=moving, type_of_transform="Rigid")


def apply_tx(mni_path: Path, in_path: Path, tx: dict, out_path: Path, interp: str = "linear"):
    fixed  = ants.image_read(str(mni_path))
    moving = ants.image_read(str(in_path))
    warped = ants.apply_transforms(
        fixed=fixed, moving=moving,
        transformlist=tx["fwdtransforms"],
        interpolator=interp,
    )
    ants.image_write(warped, str(out_path))


def warp_mask(mni_path: Path, mask_path: Path, tx: dict, out_mask_path: Path):
    fixed  = ants.image_read(str(mni_path))
    moving = ants.image_read(str(mask_path))
    warped = ants.apply_transforms(
        fixed=fixed, moving=moving,
        transformlist=tx["fwdtransforms"],
        interpolator="nearestNeighbor",
    )
    ants.image_write(warped, str(out_mask_path))


# ---------------------------------------------------------------------------
# Intensity normalisation
# ---------------------------------------------------------------------------

def norm_in_mask(
    in_path: Path,
    mask_path: Path,
    out_path: Path,
    clip_low: float = 0.5,
    clip_high: float = 99.5,
):
    """Percentile-clip and normalise voxels inside the brain mask to [0, 1].

    Voxels outside the mask are set to 0.  The result overwrites out_path.
    """
    img  = nib.load(str(in_path))
    arr  = img.get_fdata()
    msk  = nib.load(str(mask_path)).get_fdata() > 0.5
    v    = arr[msk]

    if v.size == 0:
        warnings.warn(f"Empty mask for {in_path.name}; normalising over full volume.")
        v   = arr
        msk = np.ones_like(arr, dtype=bool)

    lo, hi = np.percentile(v, [clip_low, clip_high])
    arr    = np.clip(arr, lo, hi)
    arr    = (arr - lo) / (hi - lo + 1e-8)
    arr[~msk] = 0.0
    nib.save(nib.Nifti1Image(arr.astype(np.float32), img.affine, img.header), str(out_path))


# ---------------------------------------------------------------------------
# Optional PNG export
# ---------------------------------------------------------------------------

def export_mid_slices_png(in_path: Path, out_dir: Path, n_slices: int = 12, size: int = 256):
    """Export axial slices around the volume midpoint as PNG files."""
    img  = nib.load(str(in_path))
    arr  = img.get_fdata()
    zmid = arr.shape[2] // 2
    z0   = max(0, zmid - n_slices // 2)
    zs   = range(z0, min(arr.shape[2], z0 + n_slices))
    paths = []
    for z in zs:
        sl  = (arr[:, :, z] * 255).astype(np.uint8)
        sl  = resize(sl, (size, size), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        p   = out_dir / f"z{z:03d}.png"
        iio.imwrite(p, sl)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Per-series pipeline
# ---------------------------------------------------------------------------

def process_series(row: dict, args, dirs: dict, mni_path: Path, device: str) -> dict:
    """Run the full preprocessing pipeline for a single MRI series.

    Args:
        row: Manifest row dict with keys patient, series_uid, nii, modality, site.
        args: Parsed argparse namespace.
        dirs: Output subdirectory paths.
        mni_path: Path to the MNI152 template.
        device: HD-BET device string ('cpu' or 'cuda:0').

    Returns:
        Result dict with status and output paths.
    """
    patient    = row["patient"]
    series_uid = row["series_uid"]
    modality   = row.get("modality", "")
    site       = row.get("site", "")
    base       = f"{safe_name(patient)}_{safe_name(series_uid)}"
    nii_in     = norm_path(row["nii"])

    brain_nii = dirs["brains"] / f"{base}.nii.gz"
    mask_nii  = dirs["masks"]  / f"{base}_mask.nii.gz"
    mni_nii   = dirs["rigid"]  / f"{base}_mni.nii.gz"
    mni_msk   = dirs["rigid"]  / f"{base}_mni_mask.nii.gz"

    if args.skip_done and mni_nii.exists() and mni_msk.exists():
        log(f"Skip (done): {base}")
        return {"patient": patient, "series_uid": series_uid, "status": "skipped", "png_count": 0}

    # 1. Skull stripping
    if args.fast_rebuild and brain_nii.exists() and mask_nii.exists() and not is_binary_nii(brain_nii):
        pass  # Reuse existing skull-stripped volumes
    else:
        run_hdbet(nii_in, brain_nii, mask_nii, device=device)

    # 2. N4 bias field correction (in-place)
    n4_bias_correct(brain_nii, mask_nii)

    # 3. Rigid registration to MNI
    tx = rigid_to_mni(mni_path, brain_nii)

    # 4. Apply transform to brain and mask
    apply_tx(mni_path, brain_nii, tx, mni_nii, interp="linear")
    warp_mask(mni_path, mask_nii, tx, mni_msk)

    # 5. Intensity normalisation (in-place)
    norm_in_mask(mni_nii, mni_msk, mni_nii)

    # 6. Optional PNG export
    png_count = 0
    if args.export_png:
        out_png_dir = dirs["png"] / safe_name(patient) / safe_name(series_uid)
        ensure_dir(out_png_dir)
        paths     = export_mid_slices_png(mni_nii, out_png_dir, args.T, args.size)
        png_count = len(paths)

    return {
        "patient":    patient,
        "series_uid": series_uid,
        "status":     "ok",
        "png_count":  png_count,
        "modality":   modality,
        "site":       site,
        "mni_path":   str(mni_nii),
        "mni_mask":   str(mni_msk),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse

    ap = argparse.ArgumentParser(description="Preprocess 3D structural MRI volumes.")
    ap.add_argument("--manifest",     type=str, default="nifti_out/manifest_.csv")
    ap.add_argument("--mni",          type=str, default="templates/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii")
    ap.add_argument("--out",          type=str, default="preproc_out")
    ap.add_argument("--start",        type=int, default=0,    help="Start index (inclusive)")
    ap.add_argument("--end",          type=int, default=None, help="End index (exclusive)")
    ap.add_argument("--device",       type=str, default="cpu", help="HD-BET device: cpu or cuda:0")
    ap.add_argument("--export-png",   action="store_true",    help="Export axial PNG slices for QC")
    ap.add_argument("--T",            type=int, default=12,   help="Number of axial slices to export")
    ap.add_argument("--size",         type=int, default=256,  help="PNG size in pixels (square)")
    ap.add_argument("--skip-done",    action="store_true",    help="Skip series already in rigid/")
    ap.add_argument("--fast-rebuild", action="store_true",    help="Reuse existing skull-stripped volumes")
    ap.add_argument("--ants-threads", type=int, default=4,   help="ANTs thread count")
    args = ap.parse_args()

    mni_path = Path(args.mni)
    if not mni_path.exists():
        sys.exit(f"MNI template not found: {mni_path}")

    out_root = Path(args.out)
    dirs = {
        "brains": out_root / "brains",
        "masks":  out_root / "masks",
        "rigid":  out_root / "rigid",
        "png":    out_root / "png",
        "logs":   out_root / "logs",
    }
    for d in dirs.values():
        ensure_dir(d)

    batch_tag = f"{os.getpid()}_{args.start}_{args.end if args.end is not None else 'end'}"
    log_csv   = dirs["logs"] / f"preprocess_log_{batch_tag}.csv"

    if not log_csv.exists():
        pd.DataFrame(
            columns=["patient", "series_uid", "status", "png_count", "modality", "site", "mni_path", "mni_mask"]
        ).to_csv(log_csv, index=False)

    atexit.register(lambda: print(f"[{time.strftime('%H:%M:%S')}] Log saved: {log_csv}", flush=True))

    df = pd.read_csv(args.manifest)
    missing = {"patient", "series_uid", "nii"} - set(df.columns)
    if missing:
        sys.exit(f"Manifest missing columns: {missing}")

    rows = df.to_dict(orient="records")
    rows = rows[args.start : (len(rows) if args.end is None else args.end)]
    log(f"Processing {len(rows)} series [{args.start}:{args.end}]")

    for i, row in enumerate(rows, start=1):
        tag = f"{row['patient']}/{row['series_uid']}"
        try:
            log(f"[{i}/{len(rows)}] {tag}")
            res = process_series(row, args, dirs, mni_path, args.device)
        except subprocess.CalledProcessError as e:
            log(f"ERROR (subprocess) {tag}: {e}")
            res = {
                "patient": row["patient"], "series_uid": row["series_uid"],
                "status": "fail_subprocess", "png_count": 0,
                "modality": row.get("modality", ""), "site": row.get("site", ""),
                "mni_path": "", "mni_mask": "",
            }
        except Exception as e:
            log(f"ERROR {tag}: {e}")
            res = {
                "patient": row["patient"], "series_uid": row["series_uid"],
                "status": f"fail:{type(e).__name__}", "png_count": 0,
                "modality": row.get("modality", ""), "site": row.get("site", ""),
                "mni_path": "", "mni_mask": "",
            }

        pd.DataFrame([res]).to_csv(log_csv, mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
