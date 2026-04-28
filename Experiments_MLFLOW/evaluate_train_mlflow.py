"""
Train-set evaluation for MLflow-trained models.

Uses the same preprocessing as MLflow training (Albumentations pad→crop to 384)
so metrics are computed consistently with how the model was trained.

Differences vs src/evaluate.py (production evaluation):
  - Dataset: Experiments_MLFLOW TrainDataset(train=False) → pad+crop 384×384
  - Tiles: unique tiles only (count field ignored, no duplication)
  - Output: output/metrics_<model_stem>.parquet  (same location, same schema)

Usage (from repo root):
    # UNet baseline
    CUDA_VISIBLE_DEVICES="" python -m Experiments_MLFLOW.evaluate_train_mlflow \\
        --model-path models/mlflow_unet_baseline_epoch200.pth

    # SegFormer (res2net encoder)
    CUDA_VISIBLE_DEVICES="" python -m Experiments_MLFLOW.evaluate_train_mlflow \\
        --model-path models/mlflow_segf_epoch200.pth

    # SegFormer MiT-B3
    CUDA_VISIBLE_DEVICES="" python -m Experiments_MLFLOW.evaluate_train_mlflow \\
        --model-path models/mlflow_segformer_mit_b3_stable_epoch200.pth

    # Force re-run, custom batch size:
    CUDA_VISIBLE_DEVICES="" python -m Experiments_MLFLOW.evaluate_train_mlflow \\
        --model-path models/mlflow_segf_epoch200.pth --force --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).parent.parent
EXPERIMENTS_DIR = Path(__file__).parent
METADATA_DIR    = EXPERIMENTS_DIR / "metadata"
OUTPUT_DIR      = ROOT / "output"

BALANCED_TILES_JSON = METADATA_DIR / "balanced_tiles.json"
DEFAULT_DATA_DIR    = EXPERIMENTS_DIR / "data" / "train_data"

# ── Imports from sibling modules ───────────────────────────────────────────────
from Experiments_MLFLOW.data.dataset import (  # noqa: E402
    _get_validation_augmentation,
    _load_image,
    _load_mask,
    _remap_mask,
    collate_pad,
    IGNORE_INDEX,
    MODEL_BANDS,
    TRAIN_MEAN,
    TRAIN_STD,
)
from src.evaluate import compute_tile_metrics   # noqa: E402
from src.model import load_model                # noqa: E402


# ── Deduplicated tile dataset ──────────────────────────────────────────────────

class UniqueTrainDataset(Dataset):
    """
    Loads unique training tiles (one entry per imagery_file, count ignored).
    Applies validation augmentation: Normalize → PadIfNeeded(384) → Crop(384).
    """

    def __init__(
        self,
        balanced_tiles_json: Path = BALANCED_TILES_JSON,
        data_dir: Path = DEFAULT_DATA_DIR,
        model_bands: list[str] = MODEL_BANDS,
        mean: list[float] = TRAIN_MEAN,
        std: list[float] = TRAIN_STD,
    ):
        self.data_dir    = Path(data_dir)
        self.model_bands = model_bands
        self.transform   = _get_validation_augmentation(mean, std)

        with open(balanced_tiles_json) as f:
            raw_tiles = json.load(f)["balanced_tiles"]

        # Deduplicate by imagery_file (ignore count)
        seen: set[str] = set()
        entries: list[dict] = []
        skipped = 0
        for tile in raw_tiles:
            key = tile["imagery_file"]
            if key in seen:
                continue
            seen.add(key)
            if not (self.data_dir / tile["imagery_file"]).exists() or \
               not (self.data_dir / tile["mask_file"]).exists():
                skipped += 1
                continue
            entries.append(tile)

        if skipped:
            print(f"[UniqueTrainDataset] {skipped} tiles skipped (not on disk)")
        print(f"[UniqueTrainDataset] {len(entries):,} unique tiles loaded")
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        entry = self.entries[i]
        img  = _load_image(self.data_dir / entry["imagery_file"], self.model_bands)
        mask = _load_mask(self.data_dir / entry["mask_file"])
        mask = _remap_mask(mask)

        result  = self.transform(image=img, mask=mask)
        img_out = result["image"]
        msk_out = result["mask"]

        image_t = torch.from_numpy(img_out.transpose(2, 0, 1))
        mask_t  = torch.from_numpy(msk_out.astype(np.int64))

        meta = {
            "imagery_file": entry["imagery_file"],
            "mask_file":    entry["mask_file"],
            "tile_idx":     i,
            "count":        entry.get("count", 1),
        }
        return image_t, mask_t, meta


# ── Evaluation loop ────────────────────────────────────────────────────────────

def run_evaluation(
    model_path: Path,
    data_dir: Path = DEFAULT_DATA_DIR,
    tiles_json: Path = BALANCED_TILES_JSON,
    batch_size: int = 4,
    force: bool = False,
    num_workers: int = 4,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path)
    model_stem = model_path.stem
    out_parquet = OUTPUT_DIR / f"metrics_{model_stem}.parquet"
    out_csv     = OUTPUT_DIR / f"metrics_{model_stem}.csv"

    if out_parquet.exists() and not force:
        print(f"[eval_train] {out_parquet.name} exists — loading from cache "
              "(use --force to re-run).")
        return pd.read_parquet(out_parquet)

    if not model_path.exists():
        print(f"[eval_train] ERROR: model not found: {model_path}")
        sys.exit(1)

    dataset = UniqueTrainDataset(
        balanced_tiles_json=tiles_json,
        data_dir=data_dir,
    )
    if len(dataset) == 0:
        print("[eval_train] ERROR: 0 tiles on disk — check --data-dir")
        sys.exit(1)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_train] Model : {model_path.name}")
    print(f"[eval_train] Device: {device}")
    model = load_model(model_path, device=device)

    rows = []
    skipped = 0
    f1_sum = 0.0

    bar = tqdm(total=len(dataset), unit="tile", dynamic_ncols=True,
               desc=f"  {model_stem[:40]}")

    with torch.no_grad():
        for images, masks, metas in loader:
            images = images.to(device)
            probs  = model(images)
            preds  = probs.argmax(dim=1)

            preds_np = preds.cpu().numpy()
            masks_np = masks.numpy()

            for i in range(images.shape[0]):
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
                f1_sum += metrics["f1_erosion"]
                bar.update(1)
                bar.set_postfix(
                    f1_ero=f"{f1_sum / max(1, len(rows)):.3f}",
                    skipped=skipped,
                    refresh=False,
                )

    bar.close()

    if not rows:
        print("[eval_train] No valid tiles — nothing saved.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"[eval_train] Saved {len(df):,} rows → {out_parquet}")

    print(f"\n── Mean metrics — {model_stem} ──")
    for col in ["f1_erosion", "f1_no_erosion", "iou_erosion",
                "precision_erosion", "recall_erosion"]:
        print(f"  {col:30s}: {df[col].mean():.4f}")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MLflow-trained model on training tiles "
                    "(uses MLflow preprocessing: pad+crop to 384×384)"
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to model .pth checkpoint")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help=f"Directory with train NPZ tiles (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--tiles-json", default=str(BALANCED_TILES_JSON),
                        help=f"Balanced tiles JSON (default: {BALANCED_TILES_JSON})")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Inference batch size (default: 4)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if output parquet exists")
    args = parser.parse_args()

    run_evaluation(
        model_path  = Path(args.model_path),
        data_dir    = Path(args.data_dir),
        tiles_json  = Path(args.tiles_json),
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        force       = args.force,
    )


if __name__ == "__main__":
    main()
