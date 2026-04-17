"""
Tests for data/dataset.py — requires NPZ files to be downloaded first.

Run after download_train.py + download_test.py:
    python Experiments_MLFLOW/tests/test_dataset.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import Experiments_MLFLOW.config as cfg
from Experiments_MLFLOW.data.dataset import TrainDataset, TestDataset, collate_pad


IGNORE_INDEX = cfg.IGNORE_INDEX


def test_train_dataset():
    if not cfg.BALANCED_TILES_JSON.exists():
        print("[SKIP] TrainDataset: balanced_tiles.json not found — run download_train.py first")
        return
    ds = TrainDataset(
        balanced_tiles_json=cfg.BALANCED_TILES_JSON,
        data_dir=cfg.TRAIN_DATA_DIR,
        train=False,  # no augmentation for deterministic test
    )
    if len(ds) == 0:
        print("[SKIP] TrainDataset: balanced_tiles.json not found or no NPZs on disk")
        return

    print(f"  TrainDataset: {len(ds):,} entries (with tile duplication)")

    img, mask, meta = ds[0]
    assert img.shape[0] == cfg.IN_CHANNELS, f"Expected {cfg.IN_CHANNELS} channels, got {img.shape}"
    assert img.ndim == 3, "Image must be 3D (C, H, W)"
    assert mask.ndim == 2, "Mask must be 2D (H, W)"
    assert img.dtype == torch.float32
    assert mask.dtype == torch.long

    valid = mask != IGNORE_INDEX
    if valid.any():
        unique = mask[valid].unique().tolist()
        assert all(v in [0, 1] for v in unique), f"Unexpected mask values: {unique}"

    print(f"  Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"  Image range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"  Mask unique (excl. ignore): {mask[valid].unique().tolist()}")
    print("[PASS] TrainDataset basic checks")


def test_test_dataset():
    ds = TestDataset(
        test_metadata_json=cfg.TEST_METADATA_JSON,
        data_dir=cfg.TEST_DATA_DIR,
    )
    if len(ds) == 0:
        print("[SKIP] TestDataset: no NPZs on disk yet")
        return

    print(f"  TestDataset: {len(ds):,} tiles")

    img, mask, meta = ds[0]
    assert img.shape[0] == cfg.IN_CHANNELS
    assert mask.ndim == 2
    assert "capture_name" in meta
    print(f"  capture_name: {meta['capture_name']}")
    print("[PASS] TestDataset basic checks")


def test_collate_pad():
    """Verify collate_pad pads to multiple of 32 and stacks correctly."""
    if not cfg.BALANCED_TILES_JSON.exists():
        print("[SKIP] collate_pad: balanced_tiles.json not found")
        return
    ds = TrainDataset(
        balanced_tiles_json=cfg.BALANCED_TILES_JSON,
        data_dir=cfg.TRAIN_DATA_DIR,
        train=False,
    )
    if len(ds) < 2:
        print("[SKIP] collate_pad: not enough tiles")
        return

    loader = DataLoader(ds, batch_size=2, collate_fn=collate_pad)
    imgs, masks, meta = next(iter(loader))

    assert imgs.ndim == 4, f"Expected 4D batch, got {imgs.ndim}D"
    assert imgs.shape[0] == 2
    assert imgs.shape[1] == cfg.IN_CHANNELS
    assert imgs.shape[2] % 32 == 0, f"H={imgs.shape[2]} not multiple of 32"
    assert imgs.shape[3] % 32 == 0, f"W={imgs.shape[3]} not multiple of 32"
    assert masks.shape == imgs.shape[:1] + imgs.shape[2:]
    print(f"  Batch shape: images={imgs.shape}, masks={masks.shape}")
    print("[PASS] collate_pad")


if __name__ == "__main__":
    print("=== Dataset tests ===")
    test_train_dataset()
    test_test_dataset()
    test_collate_pad()
    print("Dataset tests done.")
