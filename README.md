# Deep Learning for Alzheimer's Detection from 3D Structural MRI

Master's thesis project. Trains and evaluates deep learning classifiers on volumetric T1-weighted MRI
from the [ADNI](https://adni.loni.usc.edu/) dataset, covering three classification tasks:
binary CN vs AD, multiclass CN vs MCI vs AD, and MCI-to-AD conversion prediction.

The pipeline includes a custom preprocessing chain, supervised 3D CNN training,
self-supervised SimCLR pretraining, knowledge distillation to a lightweight student model,
and 3D Grad-CAM explainability.

---

## Results

### Binary classification: CN vs AD

| Model | Accuracy | AUC |
|---|---|---|
| ResNet34 + SE/GARB | 91.21% | 0.9634 |
| ResNet34 + SE/GARB (SSL pretrained) | **92.34%** | **0.9507** |
| Tiny CNN (distilled, ADNI test set) | 91.30% | 0.9773 |
| Tiny CNN (OASIS generalisation) | 83.62% | 0.9155 |

### Multiclass classification: CN vs MCI vs AD

| Model | Accuracy | Macro F1 |
|---|---|---|
| ResNet34 + SE/GARB | **61.15%** | **0.6192** |

### MCI conversion prediction

| Model | Accuracy | AUC |
|---|---|---|
| MRI + Clinical features fusion | **83.38%** | **0.904** |

---

## Project structure

```
preprocessing/
    convert_to_nifti.py      DICOM → NIfTI via dcm2niix
    preprocess.py            HD-BET + N4 + ANTs rigid MNI + intensity normalisation
    build_dataset_index.py   Aggregate preprocessed cohorts into a CSV index

models/
    cnn.py                   ResNet, DenseNet, ViT classifiers (MONAI backbone)
    resnet_se_garb.py        ResNet34 + Squeeze-and-Excitation + GARB attention
    hybrid.py                DenseNet + Transformer hybrid
    student.py               Lightweight student models (Tiny CNN, SmallResNet)
    simclr.py                SimCLR backbone and classifier head

training/
    dataset.py               ADNIDataset3D — NIfTI loader with label mapping
    transforms.py            3D augmentation pipelines (train / eval)
    splits.py                Patient-level train/val/test split (no scan leakage)
    train.py                 Supervised training loop
    train_distillation.py    Knowledge distillation (Hinton et al., 2015)

ssl/
    losses.py                NT-Xent contrastive loss
    pretrain.py              SimCLR pretraining loop
    finetune.py              Fine-tuning pretrained encoder on labelled data

evaluation/
    evaluate.py              Supervised model evaluation (ROC, confusion matrix)
    evaluate_ssl.py          SSL fine-tuned model evaluation
    evaluate_distillation.py Teacher / student / baseline comparison

xai/
    activations.py           Forward-hook activation and gradient extractor
    base_cam.py              Abstract 3D Grad-CAM base class
    grad_cam.py              3D Grad-CAM (Selvaraju et al., 2017)
    targets.py               CAM target classes
    generate_visualizations.py  Grad-CAM visualisation pipeline for the test set
```

---

## Setup

```bash
pip install -r requirements.txt
```

External tools required for preprocessing:

- [HD-BET](https://github.com/MIC-DKFZ/HD-BET) — skull stripping
- [dcm2niix](https://github.com/rordenlab/dcm2niix) — DICOM to NIfTI conversion
- MNI152 template — e.g. `icbm_avg_152_t1_tal_nlin_symmetric_VI.nii`

---

## Usage

### Preprocessing

```bash
# Step 1: convert DICOM to NIfTI
python preprocessing/convert_to_nifti.py

# Step 2: skull strip, register, normalise
python preprocessing/preprocess.py \
    --manifest nifti_out/manifest_.csv \
    --mni templates/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii \
    --out preproc_out \
    --device cuda:0 --skip-done

# Step 3: build unified dataset index
python preprocessing/build_dataset_index.py
```

### Patient-level split

```bash
# Binary (CN vs AD)
python training/splits.py --mode supervised \
    --input dataset_index_3d_full.csv \
    --out data \
    --train-ratio 0.7 --val-ratio 0.15

# With MCI (for SSL pretraining unlabelled pool)
python training/splits.py --mode ssl \
    --input dataset_index_3d_full.csv \
    --out data
```

### Training

```bash
# Supervised
python training/train.py

# SimCLR pretraining
python ssl/pretrain.py

# Fine-tune on labelled data
python ssl/finetune.py

# Knowledge distillation
python training/train_distillation.py
```

### Evaluation

```bash
python evaluation/evaluate.py
python evaluation/evaluate_ssl.py
python evaluation/evaluate_distillation.py
```

### Grad-CAM visualisations

```bash
python xai/generate_visualizations.py
```

---

## Methods

**Preprocessing.** Raw DICOM series are converted to NIfTI, skull-stripped with HD-BET,
bias-corrected with ANTs N4, rigidly registered to MNI152 space, and intensity-normalised
via percentile clipping to [0, 1] within the brain mask.

**Supervised training.** The primary classifier is a ResNet34 augmented with
Squeeze-and-Excitation (SE) channel attention and Global-Aware Residual Blocks (GARB),
which combine 3×3×3 and 5×5×5 convolutions in a dual-branch residual structure.
Training uses CosineAnnealingWarmRestarts with a weighted cross-entropy loss.

**Self-supervised pretraining.** SimCLR contrastive pretraining is applied to the
full scan pool (including unlabelled MCI subjects) using the NT-Xent loss.  The
pretrained encoder is then fine-tuned on the CN/AD subset.

**Knowledge distillation.** A Tiny CNN student is trained to match the soft output
distribution of the ResNet34 teacher (temperature-scaled KL divergence + cross-entropy).
The distilled student retains classification performance while reducing parameter count
by over 95%.

**Explainability.** 3D Grad-CAM adapts the standard Grad-CAM algorithm to volumetric
feature maps by averaging gradients over all three spatial dimensions (D, H, W).
Axial slices with the highest summed attention are visualised with heatmap overlays.

---

## Data access

ADNI data requires registration at [adni.loni.usc.edu](https://adni.loni.usc.edu/).
This repository contains no patient data.
