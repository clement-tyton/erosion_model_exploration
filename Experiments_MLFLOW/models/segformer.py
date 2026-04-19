"""SegFormer model builder using segmentation_models_pytorch (>=0.4.0)."""

import segmentation_models_pytorch as smp


def build_segformer(
    in_channels: int = 4,
    num_classes: int = 2,
    encoder_name: str = "timm-res2net101_26w_4s",
    encoder_depth: int = 5,
    encoder_weights: str = "imagenet",
) -> smp.Segformer:
    """
    Build SegFormer with the given encoder (same signature as build_unet).

    encoder_name: any SMP-compatible encoder, e.g.:
        - "timm-res2net101_26w_4s"  (same backbone as UNet → fair comparison)
        - "mit_b2"                  (native MiT encoder, hierarchical)
    activation=None → raw logits.
    SMP SegFormer bilinearly upsamples output to input resolution internally.
    """
    return smp.Segformer(
        encoder_name=encoder_name,
        encoder_depth=encoder_depth,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,
    )
