"""
SimCLR model and classifier utilities for self-supervised pretraining.

The SimCLR framework (Chen et al., 2020) trains an encoder to produce similar
representations for two augmented views of the same volume. After pretraining,
the encoder is reused as an initialisation for supervised fine-tuning.

Supported encoders: DenseNet-121, ResNet-18/34/50/101/152.
"""

import torch
import torch.nn as nn
from typing import Tuple

from monai.networks.nets import DenseNet121, ResNet


def _build_resnet(spatial_dims, in_channels, out_channels, depth):
    """Construct a MONAI ResNet with the channel config used across this project."""
    depth_configs = {
        18:  ((2, 2, 2, 2), "basic"),
        34:  ((3, 4, 6, 3), "basic"),
        50:  ((3, 4, 6, 3), "bottleneck"),
        101: ((3, 4, 23, 3), "bottleneck"),
        152: ((3, 8, 36, 3), "bottleneck"),
    }
    if depth not in depth_configs:
        raise ValueError(f"Unsupported ResNet depth: {depth}. Use 18, 34, 50, 101, or 152.")

    layers, block = depth_configs[depth]
    return ResNet(
        spatial_dims=spatial_dims,
        block=block,
        layers=layers,
        block_inplanes=(32, 64, 128, 256),
        n_input_channels=in_channels,
        num_classes=out_channels,
        norm="batch",
        conv1_t_size=5,
        conv1_t_stride=1,
    )


class SimCLRModel(nn.Module):
    """
    SimCLR model with configurable encoder and a 2-layer projection head.

    During pretraining, the projection head maps encoder features into a
    lower-dimensional space where the NT-Xent contrastive loss is applied.
    After pretraining, only the encoder is retained for downstream tasks.

    Args:
        spatial_dims: 2 for 2D, 3 for 3D data.
        in_channels: Number of input channels (1 for greyscale MRI).
        encoder_out_dim: Encoder output dimension.
        projection_dim: Projection head output dimension.
        hidden_dim: Projection head hidden dimension.
        encoder_type: 'densenet121' or 'resnet'.
        resnet_depth: ResNet depth if encoder_type is 'resnet'.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        encoder_out_dim: int = 1024,
        projection_dim: int = 128,
        hidden_dim: int = 512,
        encoder_type: str = "densenet121",
        resnet_depth: int = 50,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.encoder_out_dim = encoder_out_dim

        enc_type = encoder_type.lower()
        if enc_type == "densenet121":
            self.encoder = DenseNet121(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=encoder_out_dim,
            )
        elif enc_type.startswith("resnet"):
            self.encoder = _build_resnet(spatial_dims, in_channels, encoder_out_dim, resnet_depth)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        self.projection_head = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, C, *spatial].

        Returns:
            features:    Encoder output [B, encoder_out_dim].
            projections: Projection head output [B, projection_dim].
        """
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections

    def get_encoder(self) -> nn.Module:
        """Return the encoder module for downstream fine-tuning."""
        return self.encoder

    def save_encoder(self, path: str):
        """
        Save encoder weights and metadata (excluding the classification head).

        The saved checkpoint is a dict with keys:
            encoder_state, encoder_type, spatial_dims, in_channels, encoder_out_dim.
        """
        encoder_state = self.encoder.state_dict()

        if self.encoder_type == "densenet121":
            remove_keys = [k for k in encoder_state if "class_layers.out" in k]
        elif self.encoder_type.startswith("resnet"):
            remove_keys = [k for k in encoder_state if "fc" in k]
        else:
            remove_keys = []

        for key in remove_keys:
            del encoder_state[key]

        torch.save(
            {
                "encoder_state": encoder_state,
                "encoder_type": self.encoder_type,
                "spatial_dims": self.spatial_dims,
                "in_channels": self.in_channels,
                "encoder_out_dim": self.encoder_out_dim,
            },
            path,
        )
        print(f"Encoder saved to {path} ({len(remove_keys)} head keys removed)")

    def load_encoder(self, path: str):
        """Load encoder weights from a checkpoint saved by save_encoder."""
        checkpoint = torch.load(path)
        encoder_state = (
            checkpoint["encoder_state"]
            if isinstance(checkpoint, dict) and "encoder_state" in checkpoint
            else checkpoint
        )
        self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Encoder loaded from {path}")


def build_simclr_model(config: dict) -> SimCLRModel:
    """
    Build a SimCLR model from a configuration dictionary.

    Expected keys (all optional, defaults shown):
        spatial_dims (3), in_channels (1), encoder_out_dim (1024),
        projection_dim (128), hidden_dim (512),
        encoder_type ('densenet121'), resnet_depth (50).
    """
    return SimCLRModel(
        spatial_dims=config.get("spatial_dims", 3),
        in_channels=config.get("in_channels", 1),
        encoder_out_dim=config.get("encoder_out_dim", 1024),
        projection_dim=config.get("projection_dim", 128),
        hidden_dim=config.get("hidden_dim", 512),
        encoder_type=config.get("encoder_type", "densenet121"),
        resnet_depth=config.get("resnet_depth", 50),
    )


def create_classifier_from_pretrained(
    pretrained_encoder_path: str,
    num_classes: int = 2,
    spatial_dims: int = 3,
    in_channels: int = 1,
    freeze_encoder: bool = False,
    encoder_type: str = "densenet121",
    resnet_depth: int = 50,
) -> nn.Module:
    """
    Build a classifier initialised with pretrained encoder weights.

    Loads a checkpoint produced by SimCLRModel.save_encoder, removes
    any incompatible classification head keys, then optionally freezes
    the encoder so only the new head is trained.

    Args:
        pretrained_encoder_path: Path to the encoder checkpoint.
        num_classes: Number of output classes for the classifier.
        spatial_dims: Spatial dimensionality.
        in_channels: Number of input channels.
        freeze_encoder: If True, freeze all parameters except the head.
        encoder_type: Encoder architecture ('densenet121' or 'resnet').
        resnet_depth: ResNet depth (used only when encoder_type='resnet').

    Returns:
        Classifier with pretrained backbone weights loaded.
    """
    checkpoint = torch.load(pretrained_encoder_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "encoder_type" in checkpoint:
        encoder_type = checkpoint["encoder_type"]
        encoder_state = checkpoint["encoder_state"]
    else:
        encoder_state = checkpoint

    enc_type = encoder_type.lower()
    if enc_type == "densenet121":
        classifier = DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_classes,
        )
        head_keys = ["class_layers.out"]
    elif enc_type.startswith("resnet"):
        classifier = _build_resnet(spatial_dims, in_channels, num_classes, resnet_depth)
        head_keys = ["fc"]
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")

    # Remove keys that belong to the classification head (shape mismatch)
    incompatible = [k for k in list(encoder_state.keys()) if any(h in k for h in head_keys)]
    for key in incompatible:
        del encoder_state[key]

    if incompatible:
        print(f"Removed {len(incompatible)} incompatible head keys from checkpoint.")

    missing, unexpected = classifier.load_state_dict(encoder_state, strict=False)
    print(f"Loaded pretrained encoder from {pretrained_encoder_path}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    if freeze_encoder:
        for name, param in classifier.named_parameters():
            if not any(h in name for h in head_keys):
                param.requires_grad = False
        print("Encoder frozen — only the classification head is trainable.")

    total = sum(p.numel() for p in classifier.parameters())
    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    return classifier
