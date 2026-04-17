"""
Tests for training/metrics.py — ConfusionMeter correctness.

Run:
    python Experiments_MLFLOW/tests/test_metrics.py
"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Experiments_MLFLOW.training.metrics import ConfusionMeter


def _make_batch(pred_channel: int, true_channel: int, h: int = 16, w: int = 16):
    """Return (logits[1,2,H,W], targets[1,H,W]) for a constant prediction."""
    logits = torch.zeros(1, 2, h, w)
    logits[:, pred_channel, :, :] = 10.0   # force argmax = pred_channel
    targets = torch.full((1, h, w), true_channel, dtype=torch.long)
    return logits, targets


def test_perfect_erosion():
    """All erosion (class 1) predicted correctly → F1_erosion = 1.0."""
    meter = ConfusionMeter()
    logits, targets = _make_batch(pred_channel=1, true_channel=1)
    meter.update(logits, targets)
    r = meter.compute()
    assert abs(r["f1_erosion"] - 1.0) < 1e-6, f"f1_erosion={r['f1_erosion']}"
    assert abs(r["f1_no_erosion"] - 0.0) < 1e-6
    print(f"[PASS] Perfect erosion: f1_erosion={r['f1_erosion']:.3f}")


def test_perfect_no_erosion():
    """All no-erosion (class 0) predicted correctly → F1_no_erosion = 1.0."""
    meter = ConfusionMeter()
    logits, targets = _make_batch(pred_channel=0, true_channel=0)
    meter.update(logits, targets)
    r = meter.compute()
    assert abs(r["f1_no_erosion"] - 1.0) < 1e-6
    print(f"[PASS] Perfect no-erosion: f1_no_erosion={r['f1_no_erosion']:.3f}")


def test_ignore_index_excluded():
    """Pixels with value 255 (IGNORE_INDEX) must not affect metrics."""
    meter = ConfusionMeter(ignore_index=255)
    # 50% of pixels are IGNORE (255); rest are class-1 predicted correctly
    targets = torch.zeros(1, 1, 8, 8, dtype=torch.long)
    targets[0, 0, :4, :] = 255   # ignore half
    targets[0, 0, 4:, :] = 1     # valid: erosion
    logits = torch.zeros(1, 2, 8, 8)
    logits[:, 1, :, :] = 10.0    # predict erosion everywhere
    meter.update(logits, targets.squeeze(1))
    r = meter.compute()
    assert abs(r["f1_erosion"] - 1.0) < 1e-6
    print(f"[PASS] Ignore index excluded: f1_erosion={r['f1_erosion']:.3f}")


def test_zero_predictions():
    """All pixels predicted as class 0 when ground truth is all class 1 → F1_erosion=0."""
    meter = ConfusionMeter()
    logits, targets = _make_batch(pred_channel=0, true_channel=1)
    meter.update(logits, targets)
    r = meter.compute()
    assert abs(r["f1_erosion"] - 0.0) < 1e-6
    assert abs(r["iou_erosion"] - 0.0) < 1e-6
    print(f"[PASS] All-wrong erosion: f1_erosion={r['f1_erosion']:.3f}")


def test_reset():
    meter = ConfusionMeter()
    logits, targets = _make_batch(pred_channel=1, true_channel=1)
    meter.update(logits, targets)
    meter.reset()
    r = meter.compute()
    assert r["f1_erosion"] == 0.0
    print("[PASS] Reset works correctly")


if __name__ == "__main__":
    print("=== Metrics tests ===")
    test_perfect_erosion()
    test_perfect_no_erosion()
    test_ignore_index_excluded()
    test_zero_predictions()
    test_reset()
    print("All metrics tests passed.")
