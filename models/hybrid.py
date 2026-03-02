"""
Hybrid CNN + Transformer model for 3D brain MRI classification.

A DenseNet-121 backbone extracts local feature maps, which are then
treated as spatial token sequences and processed by a small Transformer
encoder to capture long-range dependencies between brain regions.

Architecture:
    1. DenseNet-121 feature extractor (removes final pooling and classifier)
    2. Spatial tokens: (B, C, D, H, W) → (B, N, C)
    3. Linear projection to transformer dimension
    4. Learnable positional encoding
    5. Transformer encoder
    6. CLS token or global average pooling → classifier

Available configurations:
    hybrid_densenet_tiny    - 128-dim transformer, 2 layers
    hybrid_densenet_small   - 256-dim transformer, 2 layers  (recommended)
    hybrid_densenet_medium  - 256-dim transformer, 4 layers
    hybrid_densenet_deep    - 256-dim transformer, 6 layers
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from monai.networks.nets import DenseNet121 as MonaiDenseNet121
    _MONAI_AVAILABLE = True
except ImportError:
    _MONAI_AVAILABLE = False


class PositionalEncoding3D(nn.Module):
    """Learnable positional encoding for a sequence of spatial tokens."""

    def __init__(self, num_tokens: int, hidden_size: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding


class TransformerEncoderBlock(nn.Module):
    """
    Standard pre-norm transformer encoder block.

    Pre-norm (normalise before attention/MLP) tends to be more stable
    for training from scratch than post-norm.

    Args:
        hidden_size: Embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim as a multiple of hidden_size.
        dropout: Dropout probability.
        attention_dropout: Dropout inside the attention mechanism.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class HybridDenseNetTransformer(nn.Module):
    """
    DenseNet-121 backbone with a Transformer encoder head.

    Args:
        num_classes: Number of output classes.
        input_size: Input volume size (D, H, W).
        transformer_hidden: Transformer embedding dimension.
        transformer_layers: Number of transformer encoder blocks.
        transformer_heads: Number of attention heads.
        mlp_ratio: MLP hidden dim ratio in the transformer.
        dropout: Dropout probability.
        use_cls_token: Use CLS token (True) or global average pooling (False).
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_size: Tuple[int, int, int] = (64, 112, 112),
        transformer_hidden: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.2,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.use_cls_token = use_cls_token
        self.transformer_hidden = transformer_hidden

        densenet = MonaiDenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
        )
        self.backbone = densenet.features
        self.cnn_features = 1024  # DenseNet-121 final feature channels

        # Compute spatial token count from a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *input_size)
            dummy_out = self.backbone(dummy)
            self.feature_shape = dummy_out.shape[2:]
            self.num_tokens = dummy_out.shape[2] * dummy_out.shape[3] * dummy_out.shape[4]

        self.proj = nn.Sequential(
            nn.Linear(self.cnn_features, transformer_hidden),
            nn.LayerNorm(transformer_hidden),
        )

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_hidden))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            total_tokens = self.num_tokens + 1
        else:
            total_tokens = self.num_tokens

        self.pos_encoding = PositionalEncoding3D(total_tokens, transformer_hidden)

        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                hidden_size=transformer_hidden,
                num_heads=transformer_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=dropout,
            )
            for _ in range(transformer_layers)
        ])

        self.norm = nn.LayerNorm(transformer_hidden)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(transformer_hidden, num_classes),
        )

        self._init_transformer_weights()

    def _init_transformer_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        features = self.backbone(x)                              # (B, 1024, D', H', W')
        tokens = features.flatten(2).transpose(1, 2)            # (B, N, 1024)
        tokens = self.proj(tokens)                               # (B, N, transformer_hidden)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        tokens = self.pos_encoding(tokens)

        for block in self.transformer_blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        out = tokens[:, 0] if self.use_cls_token else tokens.mean(dim=1)

        return self.classifier(out)


HYBRID_CONFIGS = {
    "hybrid_densenet_tiny": {
        "transformer_hidden": 128, "transformer_layers": 2,
        "transformer_heads": 4, "mlp_ratio": 2.0,
    },
    "hybrid_densenet_small": {
        "transformer_hidden": 256, "transformer_layers": 2,
        "transformer_heads": 4, "mlp_ratio": 4.0,
    },
    "hybrid_densenet_medium": {
        "transformer_hidden": 256, "transformer_layers": 4,
        "transformer_heads": 8, "mlp_ratio": 4.0,
    },
    "hybrid_densenet_deep": {
        "transformer_hidden": 256, "transformer_layers": 6,
        "transformer_heads": 8, "mlp_ratio": 4.0,
    },
}


def build_hybrid_3d(
    model_name: str,
    num_classes: int = 2,
    dropout: float = 0.2,
    input_size: Tuple[int, int, int] = (64, 112, 112),
    use_cls_token: bool = True,
) -> nn.Module:
    """
    Build a hybrid DenseNet + Transformer model.

    Args:
        model_name: Configuration name (see HYBRID_CONFIGS).
        num_classes: Number of output classes.
        dropout: Dropout probability.
        input_size: Input spatial dimensions (D, H, W).
        use_cls_token: Use CLS token or global average pooling.

    Returns:
        HybridDenseNetTransformer model.
    """
    if not _MONAI_AVAILABLE:
        raise ImportError("MONAI is required. Install with: pip install monai")

    name = model_name.lower()
    if name not in HYBRID_CONFIGS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(HYBRID_CONFIGS.keys())}"
        )

    cfg = HYBRID_CONFIGS[name]
    return HybridDenseNetTransformer(
        num_classes=num_classes,
        input_size=input_size,
        transformer_hidden=cfg["transformer_hidden"],
        transformer_layers=cfg["transformer_layers"],
        transformer_heads=cfg["transformer_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=dropout,
        use_cls_token=use_cls_token,
    )


if __name__ == "__main__":
    input_size = (64, 112, 112)
    x = torch.randn(2, 1, *input_size)

    print(f"{'Model':<25} {'Parameters':>13}  {'Output Shape'}")
    print("-" * 55)
    for key in HYBRID_CONFIGS:
        model = build_hybrid_3d(key, num_classes=2, input_size=input_size)
        model.eval()
        with torch.no_grad():
            out = model(x)
        total = sum(p.numel() for p in model.parameters())
        print(f"{key:<25} {total:>13,}  {tuple(out.shape)}")
