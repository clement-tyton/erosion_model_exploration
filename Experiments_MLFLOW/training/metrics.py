"""
Confusion-matrix based metrics for erosion segmentation.

Two classes:

ConfusionMeter
    Streaming global confusion matrix — for training epochs.
    Accumulates pixel-level TP/FP/FN across batches, computes epoch-end metrics.

TileMetricsCollector
    Per-tile confusion matrix + global aggregation — for test-set evaluation.
    One row per tile in the output DataFrame.
    Saves to Parquet for dashboard / leaderboard consumption.

Class mapping (channel indices):
    0 = no-erosion  (raw label 1)
    1 = erosion     (raw label 14)
    255 = IGNORE    (background — excluded from everything)

Erosion-focused metrics (erosion = class 1 = channel 1):
    precision_erosion  = TP / (TP + FP)   — of all pixels predicted erosion, how many are correct
    recall_erosion     = TP / (TP + FN)   — of all true erosion pixels, how many are found
    f1_erosion         = harmonic mean of precision and recall
    iou_erosion        = TP / (TP + FP + FN)  — pixel-wise Jaccard
    false_alarm_rate   = FP / (FP + TN)   — erosion predicted but wasn't (false positives over negatives)
    miss_rate          = FN / (TP + FN)   — same as 1 - recall; erosion missed
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

IGNORE_INDEX = 255
N_CLASSES = 2


# ── Low-level helper ──────────────────────────────────────────────────────────

def _confusion_matrix(
    pred: np.ndarray,   # (H*W,) int
    true: np.ndarray,   # (H*W,) int
    n_classes: int = N_CLASSES,
) -> np.ndarray:
    """Pixel-wise confusion matrix, IGNORE_INDEX excluded. Shape: (n_classes, n_classes)."""
    valid = true != IGNORE_INDEX
    p = pred[valid]
    t = true[valid]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for tc in range(n_classes):
        for pc in range(n_classes):
            cm[tc, pc] += int(np.sum((t == tc) & (p == pc)))
    return cm


def _metrics_from_cm(cm: np.ndarray, class_idx: int) -> dict[str, float]:
    """
    Per-class pixel-wise metrics from a confusion matrix.
    Returns precision, recall, f1, iou, and erosion-specific extras.
    All values are 0.0 when undefined (no positive samples).
    """
    tp = int(cm[class_idx, class_idx])
    fp = int(cm[:, class_idx].sum()) - tp
    fn = int(cm[class_idx, :].sum()) - tp
    tn = int(cm.sum()) - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    # Erosion-specific: false alarm rate = FP/(FP+TN), miss rate = FN/(TP+FN)
    false_alarm = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    miss_rate   = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision":   float(precision),
        "recall":      float(recall),
        "f1":          float(f1),
        "iou":         float(iou),
        "false_alarm": float(false_alarm),   # FP rate — over-detection
        "miss_rate":   float(miss_rate),     # 1-recall — under-detection
        "support":     int(cm[class_idx, :].sum()),
    }


def _full_metrics(cm: np.ndarray) -> dict[str, float]:
    """
    Compute all metrics from a (2,2) confusion matrix.
    Returns flat dict with *_erosion and *_no_erosion suffixes.
    """
    no_erosion = _metrics_from_cm(cm, class_idx=0)
    erosion    = _metrics_from_cm(cm, class_idx=1)

    total = no_erosion["support"] + erosion["support"]
    w0 = no_erosion["support"] / total if total > 0 else 0.5
    w1 = erosion["support"]    / total if total > 0 else 0.5

    return {
        # ── Erosion class (primary focus) ──────────────────────────────────
        "precision_erosion":   erosion["precision"],
        "recall_erosion":      erosion["recall"],
        "f1_erosion":          erosion["f1"],
        "iou_erosion":         erosion["iou"],
        "false_alarm_erosion": erosion["false_alarm"],
        "miss_rate_erosion":   erosion["miss_rate"],
        "tp_erosion":          erosion["tp"],
        "fp_erosion":          erosion["fp"],
        "fn_erosion":          erosion["fn"],
        "tn_erosion":          erosion["tn"],
        "n_erosion_pixels":    erosion["support"],
        # ── No-erosion class ───────────────────────────────────────────────
        "precision_no_erosion":   no_erosion["precision"],
        "recall_no_erosion":      no_erosion["recall"],
        "f1_no_erosion":          no_erosion["f1"],
        "iou_no_erosion":         no_erosion["iou"],
        "tp_no_erosion":          no_erosion["tp"],
        "fp_no_erosion":          no_erosion["fp"],
        "fn_no_erosion":          no_erosion["fn"],
        "tn_no_erosion":          no_erosion["tn"],
        "n_no_erosion_pixels":    no_erosion["support"],
        # ── Aggregate ──────────────────────────────────────────────────────
        "mean_iou":    (erosion["iou"] + no_erosion["iou"]) / 2,
        "weighted_f1":  w0 * no_erosion["f1"] + w1 * erosion["f1"],
        "mean_f1":     (erosion["f1"] + no_erosion["f1"]) / 2,
    }


# ── ConfusionMeter (training) ──────────────────────────────────────────────────

class ConfusionMeter:
    """
    Streaming global confusion matrix for training epochs.
    Accumulates across batches, computes epoch-end metrics, resets between epochs.
    """

    def __init__(self, n_classes: int = N_CLASSES, ignore_index: int = IGNORE_INDEX):
        self.n_classes    = n_classes
        self.ignore_index = ignore_index
        self._cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    def reset(self) -> None:
        self._cm[:] = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """logits: (B,C,H,W) raw, targets: (B,H,W) long with IGNORE_INDEX."""
        preds   = logits.argmax(dim=1).cpu().numpy().ravel()
        targets = targets.cpu().numpy().ravel()
        self._cm += _confusion_matrix(preds, targets, self.n_classes)

    def compute(self) -> dict[str, float]:
        return _full_metrics(self._cm)

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self._cm.copy()


# ── TileMetricsCollector (evaluation) ─────────────────────────────────────────

class TileMetricsCollector:
    """
    Per-tile confusion matrix accumulator for test-set evaluation.

    Usage:
        collector = TileMetricsCollector()
        for images, masks, meta in test_loader:
            logits = model(images)
            collector.add_batch(logits, masks, meta)

        df  = collector.to_dataframe()   # per-tile metrics
        agg = collector.compute_global() # aggregate metrics
        collector.save(path)             # save parquet
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        ignore_index: int = IGNORE_INDEX,
        model_name: str = "",
        epoch: int = 0,
    ):
        self.n_classes    = n_classes
        self.ignore_index = ignore_index
        self.model_name   = model_name
        self.epoch        = epoch
        self._rows: list[dict[str, Any]] = []
        self._global_cm   = np.zeros((n_classes, n_classes), dtype=np.int64)

    def reset(self) -> None:
        self._rows.clear()
        self._global_cm[:] = 0

    @torch.no_grad()
    def add_batch(
        self,
        logits: torch.Tensor,    # (B,C,H,W)
        targets: torch.Tensor,   # (B,H,W)
        meta: dict[str, list],   # collated meta from DataLoader
    ) -> None:
        """Process one batch — stores one row per tile."""
        preds_batch   = logits.argmax(dim=1).cpu().numpy()   # (B,H,W)
        targets_batch = targets.cpu().numpy()                 # (B,H,W)

        B = preds_batch.shape[0]
        for i in range(B):
            pred = preds_batch[i].ravel()
            true = targets_batch[i].ravel()

            tile_cm = _confusion_matrix(pred, true, self.n_classes)
            self._global_cm += tile_cm

            n_ignore = int(np.sum(true == self.ignore_index))
            metrics  = _full_metrics(tile_cm)

            row: dict[str, Any] = {
                "model_name":    self.model_name,
                "epoch":         self.epoch,
                "imagery_file":  meta["imagery_file"][i],
                "mask_file":     meta["mask_file"][i],
                "capture_id":    meta.get("capture_id",   [""] * B)[i],
                "capture_name":  meta.get("capture_name", [""] * B)[i],
                "n_ignore_pixels": n_ignore,
                **metrics,
            }
            self._rows.append(row)

    def compute_global(self) -> dict[str, float]:
        """Aggregate metrics over all tiles (pixel-level, not tile-average)."""
        return _full_metrics(self._global_cm)

    def compute_tile_average(self) -> dict[str, float]:
        """Macro-average per-tile metrics (mean of per-tile scores)."""
        if not self._rows:
            return {}
        df = self.to_dataframe()
        metric_cols = [
            "precision_erosion", "recall_erosion", "f1_erosion", "iou_erosion",
            "false_alarm_erosion", "miss_rate_erosion",
            "precision_no_erosion", "recall_no_erosion", "f1_no_erosion", "iou_no_erosion",
            "mean_iou", "weighted_f1", "mean_f1",
        ]
        return {f"tile_mean_{k}": float(df[k].mean()) for k in metric_cols if k in df.columns}

    def to_dataframe(self) -> pd.DataFrame:
        """One row per tile with all metrics."""
        return pd.DataFrame(self._rows)

    @property
    def global_confusion_matrix(self) -> np.ndarray:
        return self._global_cm.copy()

    def save(self, path: str | Path, format: str = "parquet") -> Path:
        """Save per-tile metrics to parquet (default) or csv."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        if format == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return path
