"""
Build a unified dataset index from multiple preprocessed ADNI scan cohorts.

For each scanner sequence cohort, locates the preprocessed MNI-registered
volumes under <root>/<folder>/rigid/ and checks whether each expected file
exists.  All cohorts are concatenated into a single CSV index.

Output columns: Subject, Image Data ID, Group, Sex, Age, Description, Path, Modality

Usage:
    python preprocessing/build_dataset_index.py
"""

import os

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_FOLDER = "T1_ADNI_3D_preprocessed"

# Maps human-readable sequence names to (metadata CSV, preprocessed folder).
SOURCES = {
    "MPRAGE": (
        "ADNI_metadata_MPRAGE.csv",
        "preproc_out_T1_MPRAGE",
    ),
    "MP-RAGE": (
        "ADNI_metadata_MP-RAGE.csv",
        "preproc_out_MP-RAGE",
    ),
    "SAG 3D MPRAGE": (
        "ADNI_metadata_SAG_3D_MPRAGE.csv",
        "preproc_out_SAG_3D_MPRAGE",
    ),
    "Sag IR-SPGR": (
        "ADNI_metadata_Sag_IR-SPGR.csv",
        "preproc_out_Sag_IR-SPGR",
    ),
    "Accelerated Sagittal MPRAGE": (
        "ADNI_metadata_Acc_Sagittal_MPRAGE.csv",
        "preproc_out_Acc_Sagittal_MPRAGE",
    ),
    "Sag IR-FSPGR": (
        "ADNI_metadata_Sag_IR-FSPGR.csv",
        "preproc_out_Sag_IR-FSPGR",
    ),
    "Accelerated Sagittal MPRAGE (MSV22)": (
        "ADNI_metadata_MPRAGE_MSV22.csv",
        "preproc_out_Acc_Sag_MPRAGE_MSV22",
    ),
    "Accelerated Sagittal MPRAGE (MSV21)": (
        "ADNI_metadata_MPRAGE_MSV21.csv",
        "preproc_out_MPRAGE_MSV21",
    ),
}

KEEP_COLS = ["Subject", "Image Data ID", "Group", "Sex", "Age", "Description", "Path", "Modality"]
OUTPUT_CSV = "dataset_index_3d_full.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_index():
    all_dfs = []

    for desc, (csv_file, folder) in SOURCES.items():
        csv_path = os.path.join(ROOT_FOLDER, folder, csv_file)
        df = pd.read_csv(csv_path)

        df["Path"] = df.apply(
            lambda r: os.path.join(
                ROOT_FOLDER,
                folder,
                "rigid",
                f"{r['Subject']}_{r['Image Data ID']}_mni.nii.gz",
            ),
            axis=1,
        )
        df["Description"] = desc

        exists_mask = df["Path"].apply(os.path.exists)
        n_total = len(df)
        df = df[exists_mask]
        print(f"{desc}: {len(df)}/{n_total} files found")

        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined[KEEP_COLS]
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved combined dataset with {len(combined)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    build_index()
