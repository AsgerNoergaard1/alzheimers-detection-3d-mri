"""
Base class for gradient-weighted Class Activation Map (CAM) methods on 3D volumes.

Adapted from the standard 2D Grad-CAM implementation to handle volumetric
(D, H, W) feature maps. Subclasses implement get_cam_weights() to define
the specific weighting scheme (e.g., Grad-CAM, Grad-CAM++).
"""

import numpy as np
import torch
from scipy.ndimage import zoom
from typing import Callable, List, Optional, Tuple

from xai.activations import ActivationsAndGradients
from xai.targets import ClassifierOutputTarget


class BaseCAM3D:
    """
    Abstract base for 3D gradient-based CAM methods.

    Args:
        model: Trained classification model.
        target_layers: List of layers whose activations and gradients are used.
        reshape_transform: Optional transform applied to feature maps (for ViT).
        compute_input_gradient: If True, enable gradients on the input tensor.
        uses_gradients: If True, perform a backward pass to collect gradients.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Optional[Callable] = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
    ):
        self.model = model.eval()
        self.target_layers = target_layers
        self.device = next(model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients

        self.activations_and_grads = ActivationsAndGradients(
            model, target_layers, reshape_transform
        )

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: list,
        activations: np.ndarray,
        grads: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement get_cam_weights.")

    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: list,
        activations: np.ndarray,
        grads: np.ndarray,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        # weights: (B, C); activations: (B, C, D, H, W)
        weighted = weights[:, :, None, None, None] * activations
        return weighted.sum(axis=1)   # (B, D, H, W)

    def forward(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[list] = None,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(c) for c in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum(target(output) for target, output in zip(targets, outputs))
            loss.backward(retain_graph=True)

        cam_per_layer = self._compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self._aggregate_multi_layers(cam_per_layer)

    def _get_target_shape(self, input_tensor: torch.Tensor) -> Tuple[int, int, int]:
        """Return (D, H, W) of the input tensor."""
        return input_tensor.size(-3), input_tensor.size(-2), input_tensor.size(-1)

    def _compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: list,
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [a.cpu().numpy() for a in self.activations_and_grads.activations]
        grads_list       = [g.cpu().numpy() for g in self.activations_and_grads.gradients]
        target_shape     = self._get_target_shape(input_tensor)

        cam_per_layer = []
        for i, target_layer in enumerate(self.target_layers):
            layer_activations = activations_list[i] if i < len(activations_list) else None
            layer_grads       = grads_list[i]       if i < len(grads_list)       else None

            cam = self.get_cam_image(
                input_tensor, target_layer, targets, layer_activations, layer_grads, eigen_smooth
            )
            cam = np.maximum(cam, 0)
            scaled = self._scale_cam_volume(cam, target_shape)
            cam_per_layer.append(scaled[:, None, :])

        return cam_per_layer

    def _scale_cam_volume(
        self, cam: np.ndarray, target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """Resize each CAM volume to match the input spatial dimensions."""
        result = []
        for volume in cam:
            volume = volume - volume.min()
            volume = volume / (volume.max() + 1e-7)
            zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
            result.append(zoom(volume, zoom_factors, order=1))
        return np.array(result, dtype=np.float32)

    def _aggregate_multi_layers(self, cam_per_layer: list) -> np.ndarray:
        """Average CAMs across all target layers and normalise to [0, 1]."""
        cam = np.concatenate(cam_per_layer, axis=1)
        cam = np.maximum(cam, 0)
        result = cam.mean(axis=1)
        for i in range(len(result)):
            result[i] = result[i] - result[i].min()
            result[i] = result[i] / (result[i].max() + 1e-7)
        return result

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: Optional[list] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        return self.forward(input_tensor, targets, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"CAM exception ({exc_type}): {exc_value}")
            return True
