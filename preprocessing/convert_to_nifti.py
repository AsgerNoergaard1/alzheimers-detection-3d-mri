"""
Convert DICOM files to NIfTI format using dcm2niix.

Expects an ADNI directory tree structured as:
    ADNI/<Subject>/SAG_3D_MPRAGE/<Date>/<ImageID>/  (DICOM slices)

Produces one .nii.gz per image series and writes a CSV manifest with
columns: patient, series_uid, nii, modality, site.

Usage:
    python preprocessing/convert_to_nifti.py
"""

import csv
import subprocess
from pathlib import Path

ROOT = Path("ADNI")       # Root of the unpacked ADNI directory
OUTN = Path("nifti_out")  # Output directory for NIfTI files and the manifest
OUTN.mkdir(parents=True, exist_ok=True)

rows = []

for subject_dir in ROOT.iterdir():
    if not subject_dir.is_dir():
        continue
    mpr = subject_dir / "SAG_3D_MPRAGE"
    if not mpr.is_dir():
        continue

    for date_dir in mpr.iterdir():
        if not date_dir.is_dir():
            continue

        for id_dir in date_dir.iterdir():
            if not id_dir.is_dir() or not id_dir.name.startswith("I"):
                continue

            image_id = id_dir.name       # e.g. I423568
            subject  = subject_dir.name  # e.g. 941_S_4376

            out_dir = OUTN / subject
            out_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir = out_dir / f"__tmp_{image_id}"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            subprocess.run(
                [
                    "dcm2niix", "-z", "y", "-b", "n",
                    "-o", str(tmp_dir), "-f", "tmp",
                    str(id_dir),
                ],
                check=True,
            )

            nii_files = list(tmp_dir.glob("*.nii.gz"))
            if len(nii_files) != 1:
                print(f"WARNING: {id_dir} produced {len(nii_files)} NIfTI files — skipping.")
                continue

            target = out_dir / f"{image_id}.nii.gz"
            nii_files[0].rename(target)
            for p in tmp_dir.glob("*"):
                p.unlink()
            tmp_dir.rmdir()

            rows.append(
                {
                    "patient":    subject,
                    "series_uid": image_id,
                    "nii":        str(target).replace("\\", "/"),
                    "modality":   "T1",
                    "site":       subject.split("_")[0] if "_" in subject else "",
                }
            )

manifest_path = OUTN / "manifest_SAG_3D_MPRAGE.csv"
with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["patient", "series_uid", "nii", "modality", "site"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {manifest_path}")
