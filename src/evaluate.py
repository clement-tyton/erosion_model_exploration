"""
Run inference on all balanced tiles and compute per-tile metrics.

Usage:
    python -m src.evaluate [--force]

Flags:
    --force   Re-run inference even if metrics.csv already exists.

Output:
    output/metrics.csv  — one row per tile with F1, IOU, precision, recall
                          for each class (no-erosion and erosion).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    DATA_DIR,
    IGNORE_INDEX,
    METRICS_CSV,
    MODEL_PATH,
    NUM_WORKERS,
    OUTPUT_DIR,
    TILES_JSON,
)

METRICS_PARQUET = OUTPUT_DIR / "metrics.parquet"
from src.dataset import TileDataset, collate_pad, filter_tiles, load_tiles_json
from src.model import load_model


# ── Per-tile metric helpers ───────────────────────────────────────────────────

def _tile_confusion(pred: np.ndarray, true: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """
    Compute confusion matrix for a single tile.
    Pixels where true == IGNORE_INDEX are excluded.

    Returns:
        cm: (n_classes, n_classes) int64 array, cm[true_c, pred_c]
    """
    valid = true != IGNORE_INDEX
    p = pred[valid]
    t = true[valid]
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for tc in range(n_classes):
        for pc in range(n_classes):
            cm[tc, pc] = int(np.sum((t == tc) & (p == pc)))
    return cm


def _metrics_from_cm(cm: np.ndarray, class_idx: int) -> dict:
    """
    Derive precision, recall, F1, IOU for one class from a confusion matrix.
    Returns zeros (not NaN) when there are no samples.
    """
    tp = cm[class_idx, class_idx]
    fp = cm[:, class_idx].sum() - tp
    fn = cm[class_idx, :].sum() - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "iou":       float(iou),
        "tp":        int(tp),
        "fp":        int(fp),
        "fn":        int(fn),
    }


def compute_tile_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute per-class metrics for a single tile.

    Args:
        pred: (H, W) int — predicted channel indices {0, 1}
        true: (H, W) int — true channel indices {0, 1, IGNORE_INDEX}

    Returns dict with keys like f1_no_erosion, f1_erosion, iou_erosion, ...
    """
    cm = _tile_confusion(pred, true, n_classes=2)

    valid = true != IGNORE_INDEX
    n_no_erosion = int((true[valid] == 0).sum())
    n_erosion    = int((true[valid] == 1).sum())

    no_erosion = _metrics_from_cm(cm, class_idx=0)
    erosion    = _metrics_from_cm(cm, class_idx=1)

    return {
        # Ground truth pixel counts
        "n_no_erosion_pixels": n_no_erosion,
        "n_erosion_pixels":    n_erosion,
        # No-erosion class (channel 0 = class label 1)
        "precision_no_erosion": no_erosion["precision"],
        "recall_no_erosion":    no_erosion["recall"],
        "f1_no_erosion":        no_erosion["f1"],
        "iou_no_erosion":       no_erosion["iou"],
        "tp_no_erosion":        no_erosion["tp"],
        "fp_no_erosion":        no_erosion["fp"],
        "fn_no_erosion":        no_erosion["fn"],
        # Erosion class (channel 1 = class label 14)
        "precision_erosion": erosion["precision"],
        "recall_erosion":    erosion["recall"],
        "f1_erosion":        erosion["f1"],
        "iou_erosion":       erosion["iou"],
        "tp_erosion":        erosion["tp"],
        "fp_erosion":        erosion["fp"],
        "fn_erosion":        erosion["fn"],
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def run_evaluation(
    force: bool = False,
    model_path: Path = MODEL_PATH,
    tiles_json_path: Path | None = None,
    data_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """
    Run inference on all tiles and save per-tile metrics to a parquet file.

    Args:
        force:           Re-run even if output parquet already exists.
        model_path:      Path to model .pth checkpoint.
        tiles_json_path: Path to the tiles JSON (overrides config TILES_JSON).
                         Defaults to TILES_JSON from config.
        data_dir:        Directory containing NPZ tiles (overrides config DATA_DIR).
                         Defaults to DATA_DIR from config.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _tiles_json = Path(tiles_json_path) if tiles_json_path else TILES_JSON
    _data_dir   = Path(data_dir)        if data_dir        else DATA_DIR

    model_stem = Path(model_path).stem
    out_parquet = OUTPUT_DIR / f"metrics_{model_stem}.parquet"
    out_csv     = OUTPUT_DIR / f"metrics_{model_stem}.csv"

    # Cache check
    if out_parquet.exists() and not force:
        print(f"[evaluate] Found existing {out_parquet} — loading from cache.")
        print("[evaluate] Use --force to re-run inference.")
        return pd.read_parquet(out_parquet)

    # ── Data ──────────────────────────────────────────────────────────────────
    if not _data_dir.exists():
        print(
            f"[evaluate] ERROR: data_dir does not exist: {_data_dir}\n"
            "Download tiles first (see models_registry.json for S3 paths)."
        )
        sys.exit(1)

    print(f"[evaluate] Loading tiles from {_tiles_json}")
    all_tiles = load_tiles_json(_tiles_json)
    tiles = filter_tiles(all_tiles)
    print(f"[evaluate] {len(all_tiles)} total tiles → {len(tiles)} have required bands")

    dataset = TileDataset(tiles, data_dir=_data_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Loading model {model_path.name} on {device}")
    model = load_model(model_path, device=device)

    # ── Inference loop ────────────────────────────────────────────────────────
    rows = []
    skipped = 0
    f1_erosion_sum = 0.0

    bar = tqdm(total=len(dataset), unit="tile", dynamic_ncols=True, desc="Evaluating")

    with torch.no_grad():
        for images, masks, metas in loader:
            images = images.to(device)
            probs  = model(images)
            preds  = probs.argmax(dim=1)

            preds_np = preds.cpu().numpy()
            masks_np = masks.numpy()

            batch_size = images.shape[0]
            for i in range(batch_size):
                metrics = compute_tile_metrics(preds_np[i], masks_np[i])

                if metrics["n_no_erosion_pixels"] + metrics["n_erosion_pixels"] == 0:
                    skipped += 1
                    bar.update(1)
                    continue

                row = {
                    "imagery_file": metas["imagery_file"][i],
                    "mask_file":    metas["mask_file"][i],
                    "tile_idx":     int(metas["tile_idx"][i]),
                    "count":        int(metas["count"][i]),
                    **metrics,
                }
                rows.append(row)
                f1_erosion_sum += metrics["f1_erosion"]
                bar.update(1)
                bar.set_postfix(
                    f1_ero=f"{f1_erosion_sum / len(rows):.3f}",
                    skipped=skipped,
                    refresh=False,
                )

    bar.close()

    if skipped:
        print(f"[evaluate] Skipped {skipped} background-only tiles")

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"[evaluate] Saved {len(df)} rows → {out_parquet} + {out_csv}")

    # keep legacy paths in sync for the default model
    if model_path == MODEL_PATH:
        df.to_parquet(METRICS_PARQUET, index=False)
        df.to_csv(METRICS_CSV, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n── Mean metrics — {model_stem} ──")
    for col in ["f1_no_erosion", "f1_erosion", "iou_no_erosion", "iou_erosion",
                "precision_erosion", "recall_erosion"]:
        print(f"  {col:30s}: {df[col].mean():.4f}")

    print("\n── Worst 10 tiles by f1_erosion ──")
    worst = df.nsmallest(10, "f1_erosion")[
        ["imagery_file", "f1_erosion", "iou_erosion", "n_erosion_pixels"]
    ]
    print(worst.to_string(index=False))

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate erosion model on all tiles")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if metrics file exists")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model .pth file (default: MODEL_PATH from config)")
    parser.add_argument("--tiles-json", type=str, default=None,
                        help="Path to tiles JSON file (default: TILES_JSON from config)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing NPZ tiles (default: DATA_DIR from config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Inference batch size (default: {BATCH_SIZE} from config)")
    args = parser.parse_args()
    model_path  = Path(args.model_path)  if args.model_path  else MODEL_PATH
    tiles_json  = Path(args.tiles_json)  if args.tiles_json  else None
    data_dir    = Path(args.data_dir)    if args.data_dir    else None
    batch_size  = args.batch_size        if args.batch_size  else BATCH_SIZE
    run_evaluation(force=args.force, model_path=model_path,
                   tiles_json_path=tiles_json, data_dir=data_dir, batch_size=batch_size)


if __name__ == "__main__":
    main()
