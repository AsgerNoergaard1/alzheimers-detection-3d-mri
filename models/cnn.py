"""
3D CNN and Vision Transformer model definitions for brain MRI classification.

Supported architectures:
    CNN:
        monai_densenet121   - DenseNet-121 baseline
        monai_resnet18      - ResNet-18 (reduced channels for medical imaging)
        monai_resnet34      - ResNet-34
        monai_resnet50      - ResNet-50

    ViT (patch size variants):
        vit_small_patch16x16x16   - Small ViT, coarse patches (196 tokens)
        vit_small_patch8x16x16    - Small ViT, medium patches (392 tokens)
        vit_small_patch8x8x8      - Small ViT, fine patches (1568 tokens)

    ViT (scale variants):
        vit_tiny_patch8x16x16       - ~5-6M params
        vit_small_patch8x16x16      - ~12-15M params
        vit_small_deep_patch8x16x16 - ~22-25M params
        vit_base_patch8x16x16       - ~86M params
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from monai.networks.nets import (
        DenseNet121 as MonaiDenseNet121,
        ResNet as MonaiResNet,
        ViT,
    )
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


class FixedResNet3D(nn.Module):
    """
    MONAI ResNet wrapper optimised for 3D brain MRI.

    Differences from the default MONAI configuration:
    - Reduced channel widths (32→64→128→256 instead of 64→128→256→512)
      to better match the relatively small ADNI dataset.
    - Smaller initial kernel (5×5×5 vs 7×7×7) and stride 1 to preserve
      spatial detail for (64, 112, 112) inputs.
    - Dropout before the final linear layer.

    Args:
        depth: ResNet depth — 18, 34, or 50.
        num_classes: Number of output classes.
        dropout: Dropout probability before the classifier.
        input_size: Spatial dimensions (D, H, W).
    """

    def __init__(
        self,
        depth: int = 18,
        num_classes: int = 2,
        dropout: float = 0.2,
        input_size: Tuple[int, int, int] = (64, 112, 112),
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

        fc_in_features = block_inplanes[-1] if block == "basic" else block_inplanes[-1] * 4
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_in_features, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu", a=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ImprovedDenseNet121(nn.Module):
    """
    Thin wrapper around MONAI DenseNet-121 that matches the ResNet training setup.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        self.densenet = MonaiDenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            dropout_prob=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)


class ViT3DClassifier(nn.Module):
    """
    Vision Transformer for 3D medical imaging.

    Wraps the MONAI ViT with support for both learnable positional encoding
    and global average pooling as an alternative to the CLS token.

    Args:
        num_classes: Number of output classes.
        img_size: Input volume size (D, H, W).
        patch_size: 3D patch size (pD, pH, pW).
        hidden_size: Transformer embedding dimension.
        mlp_dim: MLP hidden dimension (typically 4× hidden_size).
        num_layers: Number of transformer blocks.
        num_heads: Number of attention heads.
        dropout_rate: Dropout probability.
        use_cls_token: Use CLS token (True) or global average pooling (False).
        pos_embed_type: Positional embedding type ('learnable' or 'sincos').
    """

    def __init__(
        self,
        num_classes: int = 2,
        img_size: Tuple[int, int, int] = (64, 112, 112),
        patch_size: Tuple[int, int, int] = (8, 16, 16),
        hidden_size: int = 384,
        mlp_dim: int = 1536,
        num_layers: int = 6,
        num_heads: int = 6,
        dropout_rate: float = 0.2,
        use_cls_token: bool = True,
        pos_embed_type: str = "learnable",
    ):
        super().__init__()

        self.use_cls_token = use_cls_token
        self.hidden_size = hidden_size

        try:
            # MONAI >= 1.3 API
            self.vit = ViT(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                proj_type="conv",
                pos_embed_type=pos_embed_type,
                classification=use_cls_token,
                num_classes=num_classes if use_cls_token else 2,
                dropout_rate=dropout_rate,
                spatial_dims=3,
                qkv_bias=True,
            )
        except TypeError:
            # MONAI < 1.3 fallback
            self.vit = ViT(
                in_channels=1,
                img_size=img_size,
                patch_size=patch_size,
                hidden_size=hidden_size,
                mlp_dim=mlp_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                pos_embed="conv",
                classification=use_cls_token,
                num_classes=num_classes if use_cls_token else 2,
                dropout_rate=dropout_rate,
                spatial_dims=3,
            )

        if not use_cls_token:
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, num_classes),
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_cls_token:
            out, _ = self.vit(x)
            return out
        else:
            _, hidden_states = self.vit(x)
            pooled = hidden_states.mean(dim=1)
            return self.classifier(pooled)


# Predefined ViT configurations for systematic experimentation.
VIT_CONFIGS = {
    # --- Patch size ablation ---
    "vit_small_patch16x16x16": {
        "patch_size": (16, 16, 16), "hidden_size": 384,
        "mlp_dim": 1536, "num_layers": 6, "num_heads": 6,
    },
    "vit_small_patch8x16x16": {
        "patch_size": (8, 16, 16), "hidden_size": 384,
        "mlp_dim": 1536, "num_layers": 6, "num_heads": 6,
    },
    "vit_small_patch8x8x8": {
        "patch_size": (8, 8, 8), "hidden_size": 384,
        "mlp_dim": 1536, "num_layers": 6, "num_heads": 6,
    },
    # --- Scale ablation ---
    "vit_tiny_patch8x16x16": {
        "patch_size": (8, 16, 16), "hidden_size": 192,
        "mlp_dim": 768, "num_layers": 6, "num_heads": 3,
    },
    "vit_small_deep_patch8x16x16": {
        "patch_size": (8, 16, 16), "hidden_size": 384,
        "mlp_dim": 1536, "num_layers": 12, "num_heads": 6,
    },
    "vit_base_patch8x16x16": {
        "patch_size": (8, 16, 16), "hidden_size": 768,
        "mlp_dim": 3072, "num_layers": 12, "num_heads": 12,
    },
    "vit_tiny_patch8x8x8": {
        "patch_size": (8, 8, 8), "hidden_size": 192,
        "mlp_dim": 768, "num_layers": 6, "num_heads": 3,
    },
    "vit_small_deep_patch16x16x16": {
        "patch_size": (16, 16, 16), "hidden_size": 384,
        "mlp_dim": 1536, "num_layers": 12, "num_heads": 6,
    },
}


def build_cnn_3d(
    model_name: str,
    num_classes: int = 2,
    dropout: float = 0.2,
    input_size: Tuple[int, int, int] = (64, 112, 112),
    use_cls_token: bool = True,
) -> nn.Module:
    """
    Build a 3D model by name.

    Args:
        model_name: Architecture identifier (see module docstring for options).
        num_classes: Number of output classes.
        dropout: Dropout rate.
        input_size: Input spatial dimensions (D, H, W).
        use_cls_token: For ViT models — use CLS token (True) or global avg pooling (False).

    Returns:
        Constructed nn.Module ready for training.
    """
    if not _MONAI_AVAILABLE:
        raise ImportError("MONAI is required. Install with: pip install monai")

    name = model_name.lower()

    if name == "monai_densenet121":
        return ImprovedDenseNet121(num_classes=num_classes, dropout=dropout)

    if name in ("monai_resnet18", "monai_resnet34", "monai_resnet50"):
        depth = int(name.replace("monai_resnet", ""))
        return FixedResNet3D(depth=depth, num_classes=num_classes, dropout=dropout, input_size=input_size)

    if name in VIT_CONFIGS:
        cfg = VIT_CONFIGS[name]
        return ViT3DClassifier(
            num_classes=num_classes,
            img_size=input_size,
            patch_size=cfg["patch_size"],
            hidden_size=cfg["hidden_size"],
            mlp_dim=cfg["mlp_dim"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            dropout_rate=dropout,
            use_cls_token=use_cls_token,
        )

    available = (
        ["monai_densenet121", "monai_resnet18", "monai_resnet34", "monai_resnet50"]
        + list(VIT_CONFIGS.keys())
    )
    raise ValueError(f"Unknown model '{model_name}'. Available: {available}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    input_size = (64, 112, 112)
    x = torch.randn(2, 1, *input_size)

    models_to_test = {
        "DenseNet121": "monai_densenet121",
        "ResNet-18": "monai_resnet18",
        "ResNet-34": "monai_resnet34",
        "ViT-Tiny": "vit_tiny_patch8x16x16",
        "ViT-Small": "vit_small_patch8x16x16",
    }

    print(f"{'Model':<20} {'Parameters':>13}  {'Output Shape'}")
    print("-" * 55)
    for display_name, key in models_to_test.items():
        model = build_cnn_3d(key, num_classes=2, input_size=input_size)
        model.eval()
        with torch.no_grad():
            out = model(x)
        total, _ = count_parameters(model)
        print(f"{display_name:<20} {total:>13,}  {tuple(out.shape)}")
