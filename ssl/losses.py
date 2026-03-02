"""
Contrastive loss functions for SimCLR pretraining.

Provides two implementations of the normalised temperature-scaled cross
entropy loss (NT-Xent, Chen et al. 2020):
    NTXentLoss           - custom implementation
    MONAIContrastiveLoss - thin wrapper around monai.losses.ContrastiveLoss

Use get_loss_function() as a factory to select between them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalised temperature-scaled cross entropy loss (NT-Xent).

    For a batch of N samples, two augmented views of each sample are produced
    (z1_i, z2_i). The positive pair for sample i is (z1_i, z2_i); all other
    2(N-1) combinations are treated as negatives.

    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning of Visual
        Representations", ICML 2020.

    Args:
        temperature: Scales the similarity logits before softmax.
        use_cosine_similarity: If True, use cosine similarity; otherwise use
            dot product (projections should be L2-normalised either way).
    """

    def __init__(self, temperature: float = 0.5, use_cosine_similarity: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(
        self,
        projections_1: torch.Tensor,
        projections_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            projections_1: [B, D] — projections from view 1.
            projections_2: [B, D] — projections from view 2.

        Returns:
            Scalar contrastive loss.
        """
        batch_size = projections_1.shape[0]

        z1 = F.normalize(projections_1, dim=1)
        z2 = F.normalize(projections_2, dim=1)
        representations = torch.cat([z1, z2], dim=0)  # [2B, D]

        if self.use_cosine_similarity:
            sim = F.cosine_similarity(
                representations.unsqueeze(1),
                representations.unsqueeze(0),
                dim=2,
            )
        else:
            sim = torch.matmul(representations, representations.T)

        sim = sim / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive indices: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size),
        ]).to(sim.device)

        return F.cross_entropy(sim, labels)


class MONAIContrastiveLoss(nn.Module):
    """
    Wrapper around MONAI's built-in ContrastiveLoss.

    Args:
        temperature: Temperature parameter passed to ContrastiveLoss.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        from monai.losses import ContrastiveLoss
        self.loss_fn = ContrastiveLoss(temperature=temperature)

    def forward(
        self,
        projections_1: torch.Tensor,
        projections_2: torch.Tensor,
    ) -> torch.Tensor:
        projections = torch.cat([projections_1, projections_2], dim=0)
        return self.loss_fn(projections)


def get_loss_function(loss_type: str = "ntxent", temperature: float = 0.5) -> nn.Module:
    """
    Factory for contrastive loss functions.

    Args:
        loss_type: 'ntxent' or 'monai'.
        temperature: Temperature parameter.

    Returns:
        Instantiated loss module.
    """
    if loss_type == "ntxent":
        return NTXentLoss(temperature=temperature)
    if loss_type == "monai":
        return MONAIContrastiveLoss(temperature=temperature)
    raise ValueError(f"Unknown loss type '{loss_type}'. Use 'ntxent' or 'monai'.")
