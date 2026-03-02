"""
Patient-level train/val/test splitting for ADNI data.

All splits are done at the patient (subject) level to prevent any scan from
the same patient appearing in both training and test sets — a common source
of optimistic bias in the AD classification literature.

Two splitting modes are provided:

    split_supervised   - CN/AD only, produces train/val/test CSVs for
                         supervised classification.
    split_with_ssl     - CN/AD split as above, plus a train_ssl CSV that
                         augments the training set with all available MCI
                         subjects for self-supervised pretraining.

Run this script directly to generate the CSV files:

    python splits.py --input dataset_index_3d.csv --output data/ --mode supervised
    python splits.py --input dataset_index_3d.csv --output data/ --mode ssl
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_STATE = 42


def _patient_split(df_supervised: pd.DataFrame):
    """
    Split a DataFrame of CN/AD scans at the patient level.

    Returns three DataFrames: train, val, test.
    """
    patient_groups = df_supervised.groupby("Subject")["Group"].first().reset_index()

    train_subjects, temp_subjects = train_test_split(
        patient_groups,
        test_size=(1 - TRAIN_RATIO),
        stratify=patient_groups["Group"],
        random_state=RANDOM_STATE,
    )

    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=TEST_RATIO / (TEST_RATIO + VAL_RATIO),
        stratify=temp_subjects["Group"],
        random_state=RANDOM_STATE,
    )

    def subset(df, subjects):
        return df[df["Subject"].isin(subjects["Subject"])]

    return subset(df_supervised, train_subjects), subset(df_supervised, val_subjects), subset(df_supervised, test_subjects)


def split_supervised(input_csv: str, output_dir: str):
    """
    Create patient-level train/val/test splits for supervised CN/AD training.

    Args:
        input_csv:  Path to the combined dataset index CSV.
        output_dir: Directory where train_3d.csv, val_3d.csv, test_3d.csv are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    df = df[df["Path"].notnull() & df["Path"].str.endswith(".nii.gz")]
    df = df[df["Group"].isin(["CN", "AD"])]

    print(f"Loaded {len(df)} CN/AD scans from {input_csv}")
    print(f"Unique patients: {df['Subject'].nunique()}")
    print(f"Label distribution:\n{df['Group'].value_counts()}\n")

    train_df, val_df, test_df = _patient_split(df)

    for name, df_sub, filename in [
        ("Train", train_df, "train_3d.csv"),
        ("Val",   val_df,   "val_3d.csv"),
        ("Test",  test_df,  "test_3d.csv"),
    ]:
        path = os.path.join(output_dir, filename)
        df_sub.to_csv(path, index=False)
        n_patients = df_sub["Subject"].nunique()
        counts = df_sub["Group"].value_counts().to_dict()
        print(f"{name:5s}: {len(df_sub):4d} scans | {n_patients:4d} patients | {counts}")

    print(f"\nCSVs saved to: {os.path.abspath(output_dir)}")


def split_with_ssl(input_csv: str, output_dir: str):
    """
    Create splits for self-supervised pretraining + supervised fine-tuning.

    In addition to the standard train/val/test CSVs (CN/AD only), produces
    train_3d_ssl.csv which combines the training CN/AD subjects with all
    available MCI subjects. MCI is included only in the SSL training set;
    val and test remain CN/AD only.

    Args:
        input_csv:  Path to the combined dataset index CSV (including MCI).
        output_dir: Directory where all split CSVs are saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_all = pd.read_csv(input_csv)
    df_all = df_all[df_all["Path"].notnull() & df_all["Path"].str.endswith(".nii.gz")]

    print(f"Loaded {len(df_all)} scans from {input_csv}")
    print(f"Full label distribution:\n{df_all['Group'].value_counts()}\n")

    df_supervised = df_all[df_all["Group"].isin(["CN", "AD"])].copy()
    df_mci = df_all[df_all["Group"] == "MCI"].copy()

    print(f"Supervised (CN/AD):  {len(df_supervised)} scans, {df_supervised['Subject'].nunique()} patients")
    print(f"SSL only (MCI):      {len(df_mci)} scans, {df_mci['Subject'].nunique()} patients\n")

    train_df, val_df, test_df = _patient_split(df_supervised)
    train_ssl_df = pd.concat([train_df, df_mci], ignore_index=True)

    outputs = [
        ("Train",     train_df,     "train_3d.csv"),
        ("Train SSL", train_ssl_df, "train_3d_ssl.csv"),
        ("Val",       val_df,       "val_3d.csv"),
        ("Test",      test_df,      "test_3d.csv"),
    ]

    for name, df_sub, filename in outputs:
        path = os.path.join(output_dir, filename)
        df_sub.to_csv(path, index=False)
        n_patients = df_sub["Subject"].nunique()
        counts = df_sub["Group"].value_counts().to_dict()
        print(f"{name:10s}: {len(df_sub):4d} scans | {n_patients:4d} patients | {counts}")

    print(f"\nCSVs saved to: {os.path.abspath(output_dir)}")
    print("  train_3d.csv     - supervised training (CN/AD)")
    print("  train_3d_ssl.csv - SSL pretraining (CN/AD/MCI)")
    print("  val_3d.csv, test_3d.csv - evaluation (CN/AD)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-level data splitting for ADNI MRI.")
    parser.add_argument("--input", required=True, help="Path to dataset index CSV.")
    parser.add_argument("--output", default="data", help="Output directory for split CSVs.")
    parser.add_argument(
        "--mode",
        choices=["supervised", "ssl"],
        default="supervised",
        help="'supervised' for CN/AD only; 'ssl' to also create an MCI-augmented SSL split.",
    )
    args = parser.parse_args()

    if args.mode == "ssl":
        split_with_ssl(args.input, args.output)
    else:
        split_supervised(args.input, args.output)
