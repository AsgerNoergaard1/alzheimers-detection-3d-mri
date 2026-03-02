"""
3D Grad-CAM implementation.

Adapts the standard Grad-CAM (Selvaraju et al., 2017) to volumetric feature
maps by averaging gradients over all three spatial dimensions (D, H, W) instead
of just H and W.
"""

import numpy as np

from xai.base_cam import BaseCAM3D


class GradCAM3D(BaseCAM3D):
    """
    Gradient-weighted Class Activation Map for 3D inputs.

    CAM weights are computed as the global average of the gradients over the
    spatial volume:  w_c = mean_{d,h,w}(∂y^c / ∂A^k_{d,h,w})

    Args:
        model: Trained 3D classification model.
        target_layers: List of layers to compute Grad-CAM for (typically the
            last convolutional layer).
        reshape_transform: Optional activation reshape function.
    """

    def __init__(self, model, target_layers, reshape_transform=None):
        super().__init__(model, target_layers, reshape_transform)

    def get_cam_weights(
        self,
        input_tensor,
        target_layer,
        target_category,
        activations,
        grads,
    ) -> np.ndarray:
        # Average gradients over depth, height, and width: (B, C, D, H, W) -> (B, C)
        return np.mean(grads, axis=(2, 3, 4))
