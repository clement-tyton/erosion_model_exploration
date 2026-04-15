"""
Load the trained UNet segmentation model.
"""

from pathlib import Path

import torch
import torch.nn as nn

from src.config import ACTIVATION, ENCODER, ENCODER_DEPTH, MODEL_PATH


def load_model(
    checkpoint_path: Path = MODEL_PATH,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Build a UNet with timm-res2net101_26w_4s encoder and load checkpoint weights.

    Args:
        checkpoint_path: Path to the .pth file.
        device: Torch device. Defaults to CUDA if available, else CPU.

    Returns:
        model in eval mode on the target device.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as e:
        raise ImportError(
            "segmentation-models-pytorch is required. "
            "Run: uv pip install segmentation-models-pytorch"
        ) from e

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,   # we load our own checkpoint
        encoder_depth=ENCODER_DEPTH,
        in_channels=4,          # RED, GREEN, BLUE, DSM_NORMALIZED
        classes=2,              # channel 0 = no-erosion, channel 1 = erosion
        activation=ACTIVATION,
    )

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Download it from S3 using the commands in HELPERS_S3.md"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        # Checkpoint is the full model object — use it directly
        checkpoint.eval()
        return checkpoint.to(device)

    # Checkpoint is a state dict (possibly nested)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)
