"""
Enhanced 3D ResNet with Squeeze-and-Excitation (SE) attention and
Global-Aware Residual Blocks (GARB).

Two enhancement strategies are supported:

    se_only     — Adds a channel SE block before the final classifier.
                  Lightweight; good starting point.
    garb_layer3 — Replaces layer3 with GARB blocks (dual-scale 3×3×3 + 5×5×5
                  convolutions with SE attention). More parameters but captures
                  multi-scale features.

Usage:
    from models.resnet_se_garb import build_enhanced_resnet_3d

    model = build_enhanced_resnet_3d("enhanced_resnet34", num_classes=2,
                                     enhancement="garb_layer3")
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from monai.networks.nets import ResNet as MonaiResNet
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


class SEBlock3D(nn.Module):
    """
    Squeeze-and-Excitation block for 3D feature maps.

    Globally pools spatial dimensions, passes through a bottleneck MLP,
    and uses the result to recalibrate per-channel responses.

    Args:
        channels: Number of input (and output) channels.
        reduction: Channel reduction ratio for the bottleneck (default: 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale


class GARB3D(nn.Module):
    """
    Global-Aware Residual Block (3D).

    Extracts features at two scales simultaneously (3×3×3 and 5×5×5),
    concatenates them, applies SE recalibration, then adds a residual
    connection. Designed as a drop-in replacement for standard ResNet blocks
    in the intermediate layers.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Spatial stride (default: 1).
        drop_prob: Dropout3d probability after SE recalibration (default: 0.1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        drop_prob: float = 0.1,
    ):
        super().__init__()

        mid_channels = out_channels // 2

        self.conv3 = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.conv5 = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=5, stride=stride, padding=2, bias=False,
        )

        self.bn = nn.BatchNorm3d(out_channels)
        self.se = SEBlock3D(out_channels, reduction=16)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(drop_prob)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)

        out = torch.cat([self.conv3(x), self.conv5(x)], dim=1)
        out = self.bn(out)
        out = self.se(out)
        out = self.drop(out)

        out = out + identity
        return self.relu(out)


class EnhancedResNet3D(nn.Module):
    """
    3D ResNet enhanced with SE attention and optional GARB blocks.

    Args:
        depth: ResNet depth — 18, 34, or 50.
        num_classes: Number of output classes.
        dropout: Dropout probability before the classifier.
        input_size: Input spatial dimensions (D, H, W).
        enhancement: 'se_only' or 'garb_layer3'.
    """

    def __init__(
        self,
        depth: int = 34,
        num_classes: int = 2,
        dropout: float = 0.2,
        input_size: Tuple[int, int, int] = (64, 112, 112),
        enhancement: str = "se_only",
    ):
        super().__init__()

        if depth == 18:
            block, layers = "basic", (2, 2, 2, 2)
        elif depth == 34:
            block, layers = "basic", (3, 4, 6, 3)
        elif depth == 50:
            block, layers = "bottleneck", (3, 4, 6, 3)
        else:
            raise ValueError(f"Depth {depth} not supported. Use 18, 34, or 50.")

        block_inplanes = (32, 64, 128, 256)
        self.enhancement = enhancement

        if enhancement == "garb_layer3":
            self._build_with_garb(block, layers, block_inplanes, num_classes, dropout)
        else:
            self._build_with_se_only(block, layers, block_inplanes, num_classes, dropout)

    def _build_with_se_only(self, block, layers, block_inplanes, num_classes, dropout):
        self.resnet = MonaiResNet(
            spatial_dims=3,
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            n_input_channels=1,
            num_classes=num_classes,
            norm="batch",
            conv1_t_size=5,
            conv1_t_stride=1,
        )

        final_features = block_inplanes[-1] if block == "basic" else block_inplanes[-1] * 4
        self.se_final = SEBlock3D(final_features, reduction=16)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_features, num_classes),
        )

    def _build_with_garb(self, block, layers, block_inplanes, num_classes, dropout):
        self.resnet = MonaiResNet(
            spatial_dims=3,
            block=block,
            layers=layers,
            block_inplanes=block_inplanes,
            n_input_channels=1,
            num_classes=num_classes,
            norm="batch",
            conv1_t_size=5,
            conv1_t_stride=1,
        )

        # Determine actual channel counts by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 1, 64, 112, 112)
            x = self.resnet.conv1(dummy)
            x = self.resnet.bn1(x)
            x = self.resnet.act(x)
            if hasattr(self.resnet, "maxpool"):
                x = self.resnet.maxpool(x)
            x = self.resnet.layer1(x)
            x = self.resnet.layer2(x)
            layer2_out_channels = x.shape[1]
            x = self.resnet.layer3(x)
            layer3_out_channels = x.shape[1]

        garb_blocks = []
        for i in range(layers[2]):
            stride = 2 if i == 0 else 1
            in_ch = layer2_out_channels if i == 0 else layer3_out_channels
            garb_blocks.append(
                GARB3D(in_ch, layer3_out_channels, stride=stride, drop_prob=dropout)
            )
        self.resnet.layer3 = nn.Sequential(*garb_blocks)

        final_features = block_inplanes[-1] if block == "basic" else block_inplanes[-1] * 4
        self.se_final = SEBlock3D(final_features, reduction=16)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(final_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act(x)

        if hasattr(self.resnet, "maxpool"):
            x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.se_final(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x


def build_enhanced_resnet_3d(
    model_name: str,
    num_classes: int = 2,
    dropout: float = 0.2,
    input_size: Tuple[int, int, int] = (64, 112, 112),
    enhancement: str = "se_only",
) -> nn.Module:
    """
    Build an enhanced 3D ResNet with SE or GARB attention.

    Args:
        model_name: One of 'enhanced_resnet18', 'enhanced_resnet34', 'enhanced_resnet50'.
        num_classes: Number of output classes.
        dropout: Dropout probability.
        input_size: Input spatial dimensions (D, H, W).
        enhancement: 'se_only' or 'garb_layer3'.

    Returns:
        EnhancedResNet3D model.
    """
    if not _MONAI_AVAILABLE:
        raise ImportError("MONAI is required. Install with: pip install monai")

    name = model_name.lower()
    if name not in ("enhanced_resnet18", "enhanced_resnet34", "enhanced_resnet50"):
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: enhanced_resnet18, enhanced_resnet34, enhanced_resnet50"
        )

    depth = int(name.replace("enhanced_resnet", ""))
    return EnhancedResNet3D(
        depth=depth,
        num_classes=num_classes,
        dropout=dropout,
        input_size=input_size,
        enhancement=enhancement,
    )


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    input_size = (64, 112, 112)
    x = torch.randn(2, 1, *input_size)

    configs = [
        ("ResNet-34 + SE only", "enhanced_resnet34", "se_only"),
        ("ResNet-34 + GARB layer3", "enhanced_resnet34", "garb_layer3"),
    ]

    print(f"{'Model':<30} {'Parameters':>13}  {'Output Shape'}")
    print("-" * 60)
    for name, key, enh in configs:
        model = build_enhanced_resnet_3d(key, num_classes=2, enhancement=enh)
        model.eval()
        with torch.no_grad():
            out = model(x)
        total, _ = count_parameters(model)
        print(f"{name:<30} {total:>13,}  {tuple(out.shape)}")
