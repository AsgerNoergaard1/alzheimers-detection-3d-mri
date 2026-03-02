"""
MONAI-based data augmentation transforms for 3D MRI training and evaluation.

All transforms operate directly on numpy arrays (no dictionary keys required),
making them compatible with the ADNIDataset3D loader.

Two transform pipelines are provided:
    get_train_transforms_3d  - spatial and intensity augmentations for training
    get_eval_transforms_3d   - deterministic preprocessing only (val/test)
"""

from monai.transforms import (
    Compose,
    CenterSpatialCrop,
    EnsureChannelFirst,
    NormalizeIntensity,
    Rand3DElastic,
    RandAdjustContrast,
    RandAffine,
    RandFlip,
    RandGaussianNoise,
    RandGaussianSmooth,
    RandRotate,
    RandScaleIntensity,
    RandShiftIntensity,
    ToTensor,
)


def get_train_transforms_3d(target_shape=(64, 112, 112)):
    """
    Augmentation pipeline used during supervised training.

    Applies spatial augmentations (flipping, rotation, affine, elastic
    deformation) followed by intensity augmentations (scale, shift, contrast,
    noise, smoothing) to simulate scanner variability and improve generalisation.

    Args:
        target_shape: (D, H, W) to crop the volume to after channel insertion.

    Returns:
        MONAI Compose transform.
    """
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        CenterSpatialCrop(roi_size=target_shape),
        NormalizeIntensity(nonzero=False, channel_wise=True),

        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate(
            prob=0.7,
            range_x=0.262,   # ±15 degrees
            range_y=0.262,
            range_z=0.262,
            mode="bilinear",
            padding_mode="border",
        ),
        RandAffine(
            prob=0.6,
            scale_range=(0.15, 0.15, 0.15),
            shear_range=(0.1, 0.1, 0.1),
            mode="bilinear",
            padding_mode="border",
        ),
        Rand3DElastic(
            prob=0.3,
            sigma_range=(5, 7),
            magnitude_range=(50, 150),
            mode="bilinear",
            padding_mode="border",
        ),

        RandScaleIntensity(factors=0.2, prob=0.7),
        RandShiftIntensity(offsets=0.15, prob=0.7),
        RandAdjustContrast(prob=0.7, gamma=(0.7, 1.3)),
        RandGaussianNoise(prob=0.3, mean=0.0, std=0.02),
        RandGaussianSmooth(
            prob=0.3,
            sigma_x=(0.5, 1.5),
            sigma_y=(0.5, 1.5),
            sigma_z=(0.5, 1.5),
        ),

        ToTensor(),
    ])


def get_eval_transforms_3d(target_shape=(64, 112, 112)):
    """
    Deterministic preprocessing for validation and test sets.

    No random augmentation — only channel insertion, cropping,
    and intensity normalisation.

    Args:
        target_shape: (D, H, W) to crop the volume to.

    Returns:
        MONAI Compose transform.
    """
    return Compose([
        EnsureChannelFirst(channel_dim="no_channel"),
        CenterSpatialCrop(roi_size=target_shape),
        NormalizeIntensity(nonzero=False, channel_wise=True),
        ToTensor(),
    ])
