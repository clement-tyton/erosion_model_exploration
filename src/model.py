"""
Load trained segmentation models (UNet or SegFormer).
"""

from pathlib import Path

import torch
import torch.nn as nn

from src.config import ACTIVATION, ENCODER, ENCODER_DEPTH, MODEL_PATH


def _load_segformer_auto(state_dict: dict, device: torch.device) -> nn.Module:
    """
    Build a SegFormer and load the given state dict.

    Tries candidate encoder / depth combinations in order until one succeeds.
    Handles both timm-res2net and native MiT encoders transparently.
    """
    import segmentation_models_pytorch as smp

    candidates = [
        ("timm-res2net101_26w_4s", 5),
        ("mit_b3", 5),
        ("mit_b3", 4),
        ("mit_b2", 5),
        ("mit_b2", 4),
        ("mit_b4", 5),
        ("mit_b4", 4),
        ("mit_b1", 5),
        ("mit_b1", 4),
        ("mit_b5", 5),
        ("mit_b5", 4),
    ]
    for encoder_name, depth in candidates:
        try:
            model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_depth=depth,
                encoder_weights=None,
                in_channels=4,
                classes=2,
                activation=ACTIVATION,
            )
            result = model.load_state_dict(state_dict, strict=False)
            # Allow extra keys in checkpoint (e.g. SMP version differences)
            # but fail if the model has keys that are missing from the checkpoint
            if result.missing_keys:
                continue
            if result.unexpected_keys:
                print(f"[load_model] SegFormer: ignoring {len(result.unexpected_keys)} "
                      f"extra checkpoint key(s): {result.unexpected_keys}")
            model.eval()
            print(f"[load_model] SegFormer loaded with encoder={encoder_name}, depth={depth}")
            return model.to(device)
        except Exception:
            continue

    raise ValueError(
        "Could not load SegFormer checkpoint — tried all known encoder configurations.\n"
        "Candidates: " + ", ".join(f"{e}(d={d})" for e, d in candidates)
    )


def load_model(
    checkpoint_path: Path = MODEL_PATH,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Load a segmentation model checkpoint (UNet or SegFormer).

    Handles three checkpoint formats:
    - Full nn.Module object (old format)
    - MLflow dict: {"arch": "unet"|"segformer", "state_dict": {...}, ...}
    - Plain state dict or {"state_dict": {...}} / {"model": {...}}

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

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Download it from S3 using the commands in HELPERS_S3.md"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        # Full model object saved directly
        checkpoint.eval()
        return checkpoint.to(device)

    # Dict checkpoint — read arch before stripping to state_dict
    arch = None
    if isinstance(checkpoint, dict):
        arch = checkpoint.get("arch")
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    if arch == "segformer":
        return _load_segformer_auto(checkpoint, device)

    # Default: UNet (arch == "unet", or no arch key for legacy checkpoints)
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        encoder_depth=ENCODER_DEPTH,
        in_channels=4,
        classes=2,
        activation=ACTIVATION,
    )
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)
