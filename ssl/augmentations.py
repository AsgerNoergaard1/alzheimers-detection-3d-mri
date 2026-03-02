"""
Augmentation pipelines for SimCLR contrastive pretraining on 3D brain MRI.
"""

from monai.transforms import (
    Compose,
    CopyItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandCoarseDropoutd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityd,
)


# ---------------------------------------------------------------------------
# Base preprocessing
# ---------------------------------------------------------------------------

def get_base_preprocessing(keys: list = ["image"], target_shape: tuple = (64, 112, 112)) -> Compose:
    """Load and standardise a preprocessed NIfTI volume.

    Assumes scans are already skull-stripped, MNI-registered, and oriented.
    Handles loading, channel formatting, intensity scaling, and spatial resizing.

    Args:
        keys: Dictionary keys for the image tensors.
        target_shape: Desired output shape (D, H, W).

    Returns:
        MONAI Compose transform.
    """
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys=keys, minv=0.0, maxv=1.0),
        Resized(keys=keys, spatial_size=target_shape, mode="trilinear"),
    ])


# ---------------------------------------------------------------------------
# Contrastive augmentation
# ---------------------------------------------------------------------------

class ContrastiveAugmentation:
    """Augmentation pipeline that produces two independent views of the same scan.

    Augmentations are intentionally stronger than those used in supervised
    training so the encoder must learn features that are invariant to
    geometric and intensity perturbations.

    Args:
        keys: Dictionary keys for the image tensors.
        rotation_range: Maximum rotation in radians per axis (~15 deg default).
        scale_range: Maximum scaling factor.
        intensity_scale: Magnitude for random intensity scaling.
        intensity_shift: Magnitude for random intensity shifting.
        contrast_gamma: Gamma range for contrast adjustment.
        noise_std: Standard deviation for Gaussian noise.
        smooth_sigma: Sigma range for Gaussian smoothing.
        dropout_holes: Number of dropout regions per view.
        dropout_size: Spatial size of each dropout region.
        prob_geometric: Probability of applying affine transform.
        prob_intensity: Probability of applying intensity transforms.
        prob_contrast: Probability of applying contrast adjustment.
        prob_noise: Probability of applying Gaussian noise.
        prob_smooth: Probability of applying Gaussian smoothing.
        prob_dropout: Probability of applying coarse dropout.
    """

    def __init__(
        self,
        keys: list = ["image"],
        rotation_range: float = 0.26,
        scale_range: float = 0.1,
        intensity_scale: float = 0.3,
        intensity_shift: float = 0.1,
        contrast_gamma: tuple = (0.7, 1.5),
        noise_std: float = 0.1,
        smooth_sigma: tuple = (0.5, 1.0),
        dropout_holes: int = 6,
        dropout_size: tuple = (8, 8, 8),
        prob_geometric: float = 0.8,
        prob_intensity: float = 0.8,
        prob_contrast: float = 0.5,
        prob_noise: float = 0.5,
        prob_smooth: float = 0.5,
        prob_dropout: float = 0.3,
    ):
        self.keys = keys
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift
        self.contrast_gamma = contrast_gamma
        self.noise_std = noise_std
        self.smooth_sigma = smooth_sigma
        self.dropout_holes = dropout_holes
        self.dropout_size = dropout_size
        self.prob_geometric = prob_geometric
        self.prob_intensity = prob_intensity
        self.prob_contrast = prob_contrast
        self.prob_noise = prob_noise
        self.prob_smooth = prob_smooth
        self.prob_dropout = prob_dropout

    def get_transforms(self) -> Compose:
        """Return the full contrastive augmentation pipeline.

        Copies the input image into two keys (view1, view2) and applies
        independent random augmentations to each.
        """
        return Compose([
            CopyItemsd(keys=self.keys, times=2, names=["view1", "view2"]),
            self._get_view_transforms("view1"),
            self._get_view_transforms("view2"),
        ])

    def _get_view_transforms(self, view_key: str) -> Compose:
        """Build the augmentation sequence for a single view."""
        return Compose([
            # Geometric
            RandAffined(
                keys=[view_key],
                prob=self.prob_geometric,
                rotate_range=(self.rotation_range,) * 3,
                scale_range=(self.scale_range,) * 3,
                mode="bilinear",
                padding_mode="border",
            ),
            # Intensity
            RandScaleIntensityd(
                keys=[view_key],
                factors=self.intensity_scale,
                prob=self.prob_intensity,
            ),
            RandShiftIntensityd(
                keys=[view_key],
                offsets=self.intensity_shift,
                prob=self.prob_intensity,
            ),
            RandAdjustContrastd(
                keys=[view_key],
                prob=self.prob_contrast,
                gamma=self.contrast_gamma,
            ),
            # Noise and smoothing
            RandGaussianNoised(
                keys=[view_key],
                prob=self.prob_noise,
                mean=0.0,
                std=self.noise_std,
            ),
            RandGaussianSmoothd(
                keys=[view_key],
                prob=self.prob_smooth,
                sigma_x=self.smooth_sigma,
                sigma_y=self.smooth_sigma,
                sigma_z=self.smooth_sigma,
            ),
            # Coarse dropout (simulates partial volume effects)
            RandCoarseDropoutd(
                keys=[view_key],
                prob=self.prob_dropout,
                holes=self.dropout_holes,
                spatial_size=self.dropout_size,
                max_holes=self.dropout_holes * 2,
                max_spatial_size=tuple(s * 2 for s in self.dropout_size),
            ),
        ])
