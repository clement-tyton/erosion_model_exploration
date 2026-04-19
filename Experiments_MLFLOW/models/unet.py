"""UNet model builder using segmentation_models_pytorch."""

import segmentation_models_pytorch as smp


def build_unet(
    in_channels: int = 4,
    num_classes: int = 2,
    encoder_name: str = "timm-res2net101_26w_4s",
    encoder_depth: int = 5,
    encoder_weights: str = "imagenet",
) -> smp.Unet:
    """
    Build UNet with res2net encoder — same architecture as the production model.

    activation=None → raw logits (CrossEntropyLoss handles softmax).
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
