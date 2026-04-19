"""Model factory for Experiments_MLFLOW."""

import torch.nn as nn

from .unet import build_unet
from .segformer import build_segformer


def build_model(
    arch: str,
    in_channels: int = 4,
    num_classes: int = 2,
    encoder_name: str = "timm-res2net101_26w_4s",
    encoder_depth: int = 5,
    encoder_weights: str = "imagenet",
) -> nn.Module:
    """
    Build a segmentation model by architecture name.

    Both UNet and SegFormer share the same encoder_name so comparisons are fair.
    arch: "unet" or "segformer"
    """
    arch = arch.lower()
    kwargs = dict(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
    )
    if arch == "unet":
        return build_unet(**kwargs)
    elif arch == "segformer":
        return build_segformer(**kwargs)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Available: 'unet', 'segformer'")
