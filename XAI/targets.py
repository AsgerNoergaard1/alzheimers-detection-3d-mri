"""
Target classes that define which model output to backpropagate through for CAM.
"""

import torch
import torch.nn.functional as F


class ClassifierOutputTarget:
    """Backpropagate through the raw logit for a specific class index."""

    def __init__(self, category: int):
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class ClassifierOutputSoftmaxTarget:
    """Backpropagate through the softmax probability for a specific class index."""

    def __init__(self, category: int):
        self.category = category

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return F.softmax(model_output, dim=0)[self.category]
        return F.softmax(model_output, dim=1)[:, self.category]
