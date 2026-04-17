"""
Tests for models/ — forward pass shape checks for UNet and SegFormer.

Run:
    python Experiments_MLFLOW/tests/test_models.py
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Experiments_MLFLOW.models import build_model


def test_unet():
    model = build_model("unet", in_channels=4, num_classes=2)
    model.eval()
    x = torch.randn(2, 4, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 256, 256), f"UNet output shape: {out.shape}"
    print(f"[PASS] UNet output: {out.shape}")


def test_segformer():
    model = build_model("segformer", in_channels=4, num_classes=2, variant="b2",
                        encoder_weights=None)
    model.eval()
    x = torch.randn(2, 4, 256, 256)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 2, 256, 256), f"SegFormer output shape: {out.shape}"
    print(f"[PASS] SegFormer output: {out.shape}")


def test_build_model_invalid():
    try:
        build_model("invalid_arch")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("[PASS] Invalid arch raises ValueError")


if __name__ == "__main__":
    print("=== Model tests ===")
    test_unet()
    test_segformer()
    test_build_model_invalid()
    print("All model tests passed.")
