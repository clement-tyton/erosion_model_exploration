"""
Test-set download + evaluation pipeline for all registered models.

Steps
-----
1. download_test_tiles  — fetch NPZ tiles from S3 → Experiments_MLFLOW/data/test_data/
2. build_test_geo       — extract WGS84 centroids → output/tiles_geo_test.parquet
3. evaluate_model_on_test — run per-tile inference → output/test_metrics_<stem>.parquet

Usage
-----
    # Download tiles + geo + evaluate all models:
    python -m DASHBOARD_FROM_TYTONAI.evaluate_test

    # Force re-evaluation of a single model:
    python -m DASHBOARD_FROM_TYTONAI.evaluate_test --force --model model_v3_split_test_epoch80.pth

    # Only download / only build geo / only evaluate
    python -m DASHBOARD_FROM_TYTONAI.evaluate_test --skip-download --skip-geo
    python -m DASHBOARD_FROM_TYTONAI.evaluate_test --skip-download --skip-geo --skip-eval
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR   = ROOT / "Experiments_MLFLOW"
METADATA_DIR      = EXPERIMENTS_DIR / "metadata"
TEST_METADATA_JSON = METADATA_DIR / "EROSION_DATASET_TEST_METADATA.json"
DEFAULT_TEST_DATA_DIR = EXPERIMENTS_DIR / "data" / "test_data"
OUTPUT_DIR        = ROOT / "output"
MODELS_DIR        = ROOT / "models"
REGISTRY_PATH     = ROOT / "models_registry.json"

# ── S3 config (loaded lazily so missing .env only fails at download time) ─────
def _s3_cfg() -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    cfg = {
        "bucket":   os.environ.get("S3_FILE_BUCKET"),
        "endpoint": os.environ.get("AWS_S3_ENDPOINT"),
        "region":   os.environ.get("AWS_REGION", "us-east-1"),
        "key":      os.environ.get("AWS_ACCESS_KEY_ID"),
        "secret":   os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "token":    os.environ.get("AWS_SESSION_TOKEN"),
    }
    if not cfg["bucket"] or not cfg["endpoint"]:
        print("ERROR: S3_FILE_BUCKET / AWS_S3_ENDPOINT not set in .env")
        sys.exit(1)
    return cfg

# ── Geo reprojection ──────────────────────────────────────────────────────────
from pyproj import Transformer as _Transformer
_TO_WGS84 = _Transformer.from_crs("EPSG:20350", "EPSG:4326", always_xy=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Download test tiles
# ═══════════════════════════════════════════════════════════════════════════════

def download_test_tiles(
    workers: int = 32,
    data_dir: Path = DEFAULT_TEST_DATA_DIR,
) -> None:
    """Download test NPZ tiles from S3 into data_dir. Skips existing files."""
    cfg = _s3_cfg()
    import boto3
    from boto3.s3.transfer import TransferConfig
    from botocore.config import Config

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(TEST_METADATA_JSON) as f:
        entries = json.load(f)

    all_files: list[str] = []
    for entry in entries:
        all_files.append(entry["imagery_file"])
        all_files.append(entry["mask_file"])
    all_files = list(dict.fromkeys(all_files))   # deduplicate, preserve order

    to_download = [f for f in all_files if not (data_dir / f).exists()]
    n_skip = len(all_files) - len(to_download)

    print(f"Test entries       : {len(entries)}")
    print(f"Total unique files : {len(all_files)}")
    print(f"Already on disk    : {n_skip} (skipping)")
    print(f"To download        : {len(to_download)} — {workers} workers\n")

    if not to_download:
        print("[download] Nothing to download.")
        return

    _local    = threading.local()
    _xfer_cfg = TransferConfig(use_threads=False)

    def _client():
        if not hasattr(_local, "c"):
            _local.c = boto3.client(
                "s3",
                endpoint_url=f"https://{cfg['endpoint']}",
                region_name=cfg["region"],
                aws_access_key_id=cfg["key"],
                aws_secret_access_key=cfg["secret"],
                aws_session_token=cfg["token"],
                config=Config(
                    retries={"max_attempts": 3, "mode": "adaptive"},
                    max_pool_connections=workers,
                ),
            )
        return _local.c

    def _dl_one(filename: str) -> tuple[str, str]:
        dest = data_dir / filename
        if dest.exists():
            return filename, "skipped"
        try:
            _client().download_file(cfg["bucket"], filename, str(dest), Config=_xfer_cfg)
            return filename, "ok"
        except Exception as e:
            return filename, str(e)

    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_dl_one, f): f for f in to_download}
        with tqdm(total=len(to_download), unit="file", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                _, status = future.result()
                if status not in ("ok", "skipped"):
                    failed.append((futures[future], status))
                    bar.set_postfix(failed=len(failed), refresh=False)
                bar.update(1)

    print(f"\n[download] Done. Failed: {len(failed)}")
    if failed:
        for fname, err in failed:
            print(f"  {fname}\n    {err}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2 — Build geographic index for test tiles
# ═══════════════════════════════════════════════════════════════════════════════

TEST_GEO_PARQUET = OUTPUT_DIR / "tiles_geo_test.parquet"


def build_test_geo(
    data_dir: Path = DEFAULT_TEST_DATA_DIR,
    force: bool = False,
    workers: int = 8,
) -> pd.DataFrame:
    """
    Extract WGS84 centroids for every test tile and save to
    output/tiles_geo_test.parquet.

    Uses geotransform + srid from the metadata JSON and opens the imagery
    NPZ only to retrieve H × W (needed to compute centroid & bbox).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)

    if TEST_GEO_PARQUET.exists() and not force:
        print(f"[build_test_geo] {TEST_GEO_PARQUET.name} already exists — skipping "
              "(use --force to rebuild).")
        return pd.read_parquet(TEST_GEO_PARQUET)

    with open(TEST_METADATA_JSON) as f:
        entries = json.load(f)

    # Detect which tiles are on disk (skip not-yet-downloaded)
    on_disk = [e for e in entries if (data_dir / e["imagery_file"]).exists()]
    print(f"[build_test_geo] {len(on_disk)} / {len(entries)} tiles on disk, "
          f"{workers} workers…")

    def _proc(entry: dict) -> dict | None:
        img_path = data_dir / entry["imagery_file"]
        try:
            npz = np.load(img_path, mmap_mode="r")
            # Find a band key to get H, W
            band_key = next(
                (k for k in npz.keys()
                 if k not in ("SRID", "GEO_TRANSFORM", "VERSION")),
                None,
            )
            if band_key is None:
                return None
            h, w = npz[band_key].shape

            # 6-element GDAL geotransform from metadata: [x0, dx, 0, y0, 0, dy]
            gt = entry["geotransform"]
            x0, dx = float(gt[0]), float(gt[1])
            y0, dy = float(gt[3]), float(gt[5])

            # Centroid in native CRS (EPSG:20350)
            x_c = x0 + (w / 2) * dx
            y_c = y0 + (h / 2) * dy

            # Bounding box
            x_min = x0;           x_max = x0 + w * dx
            y_max = y0;           y_min = y0 + h * dy   # dy < 0 ⇒ y_min < y_max

            lon,     lat     = _TO_WGS84.transform(x_c,   y_c)
            lon_min, lat_min = _TO_WGS84.transform(x_min, y_min)
            lon_max, lat_max = _TO_WGS84.transform(x_max, y_max)

            return {
                "imagery_file": entry["imagery_file"],
                "capture_id":   entry.get("capture_id", ""),
                "capture_name": entry.get("capture_name", ""),
                "epsg":         entry.get("srid", 20350),
                "pixel_size_m": abs(dx),
                "width_px":     w,
                "height_px":    h,
                "lat":          lat,
                "lon":          lon,
                "lat_min":      lat_min,
                "lat_max":      lat_max,
                "lon_min":      lon_min,
                "lon_max":      lon_max,
            }
        except Exception as e:
            print(f"[WARN] geo {entry['imagery_file']}: {e}")
            return None

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_proc, e): e for e in on_disk}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="tile"):
            result = fut.result()
            if result:
                rows.append(result)

    df = pd.DataFrame(rows)
    df.to_parquet(TEST_GEO_PARQUET, index=False)
    print(f"[build_test_geo] Saved {len(df):,} rows → {TEST_GEO_PARQUET}")
    if not df.empty:
        print(f"  Lat range: {df['lat'].min():.4f} → {df['lat'].max():.4f}")
        print(f"  Lon range: {df['lon'].min():.4f} → {df['lon'].max():.4f}")
        captures = df["capture_name"].value_counts()
        print(f"  Captures: {captures.to_dict()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3 — Evaluate one model on the test set
# ═══════════════════════════════════════════════════════════════════════════════

def test_metrics_path(model_stem: str) -> Path:
    return OUTPUT_DIR / f"test_metrics_{model_stem}.parquet"


def evaluate_model_on_test(
    model_path: Path,
    data_dir: Path = DEFAULT_TEST_DATA_DIR,
    force: bool = False,
    batch_size: int = 4,
) -> pd.DataFrame:
    """
    Run per-tile inference on the test set for one model.

    Returns a DataFrame with the same columns as the train metrics parquet.
    Saves to output/test_metrics_<model_stem>.parquet + .csv
    """
    from src.evaluate import compute_tile_metrics
    from src.model import load_model
    from Experiments_MLFLOW.data.dataset import TestDataset, collate_pad

    model_path  = Path(model_path)
    model_stem  = model_path.stem
    out_parquet = test_metrics_path(model_stem)
    out_csv     = OUTPUT_DIR / f"test_metrics_{model_stem}.csv"

    if out_parquet.exists() and not force:
        print(f"[eval_test] {out_parquet.name} exists — loading from cache "
              "(use --force to re-run).")
        return pd.read_parquet(out_parquet)

    if not model_path.exists():
        print(f"[eval_test] ERROR: model not found: {model_path}")
        return pd.DataFrame()

    data_dir = Path(data_dir)
    if not data_dir.exists() or not any(data_dir.glob("*.npz")):
        print(f"[eval_test] ERROR: no tiles in {data_dir}. "
              "Run --skip-eval first to download tiles.")
        return pd.DataFrame()

    print(f"\n[eval_test] Model : {model_path.name}")

    dataset = TestDataset(
        test_metadata_json=TEST_METADATA_JSON,
        data_dir=data_dir,
    )
    if len(dataset) == 0:
        print("[eval_test] WARNING: 0 tiles on disk — skipping.")
        return pd.DataFrame()

    print(f"[eval_test] Tiles : {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_pad,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval_test] Device: {device}")
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
                tile_metrics = compute_tile_metrics(preds_np[i], masks_np[i])

                if (tile_metrics["n_no_erosion_pixels"]
                        + tile_metrics["n_erosion_pixels"]) == 0:
                    skipped += 1
                    bar.update(1)
                    continue

                row = {
                    "imagery_file": metas["imagery_file"][i],
                    "mask_file":    metas["mask_file"][i],
                    "tile_idx":     int(metas["tile_idx"][i]),
                    "capture_id":   metas.get("capture_id", [""])[i],
                    "capture_name": metas.get("capture_name", [""])[i],
                    **tile_metrics,
                }
                rows.append(row)
                f1_sum += tile_metrics["f1_erosion"]
                bar.update(1)
                bar.set_postfix(
                    f1_ero=f"{f1_sum / max(1, len(rows)):.3f}",
                    skipped=skipped,
                    refresh=False,
                )

    bar.close()

    if not rows:
        print("[eval_test] No valid tiles — nothing saved.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    print(f"[eval_test] Saved {len(df):,} rows → {out_parquet}")

    # Quick summary
    ero = df[df["n_erosion_pixels"] > 0]
    if not ero.empty:
        print(f"  Mean F1 erosion (non-zero tiles): {ero['f1_erosion'].mean():.4f}")
        tp = df["tp_erosion"].sum()
        fp = df["fp_erosion"].sum()
        fn = df["fn_erosion"].sum()
        d  = 2 * tp + fp + fn
        print(f"  Global F1 erosion               : {2*tp/d:.4f}" if d > 0 else "  Global F1: N/A")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download test tiles and evaluate all registered models on the test set."
    )
    parser.add_argument("--workers",       type=int,  default=32,
                        help="S3 download workers (default: 32)")
    parser.add_argument("--force",         action="store_true",
                        help="Re-run even if outputs already exist")
    parser.add_argument("--model",         type=str,  default=None,
                        help="Evaluate only this model filename (e.g. model_v3_split_test_epoch80.pth)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip tile download step")
    parser.add_argument("--skip-geo",      action="store_true",
                        help="Skip geo index building step")
    parser.add_argument("--skip-eval",     action="store_true",
                        help="Skip model evaluation step")
    args = parser.parse_args()

    # ── Step 1: download ──────────────────────────────────────────────────────
    if not args.skip_download:
        print("\n══ Step 1 / 3 — Download test tiles ══")
        download_test_tiles(workers=args.workers)
    else:
        print("[skip] Download step skipped.")

    # ── Step 2: geo ───────────────────────────────────────────────────────────
    if not args.skip_geo:
        print("\n══ Step 2 / 3 — Build geo index ══")
        build_test_geo(force=args.force)
    else:
        print("[skip] Geo step skipped.")

    # ── Step 3: evaluate ──────────────────────────────────────────────────────
    if args.skip_eval:
        print("[skip] Evaluation step skipped.")
        return

    if not REGISTRY_PATH.exists():
        print(f"ERROR: {REGISTRY_PATH} not found.")
        sys.exit(1)

    registry = json.loads(REGISTRY_PATH.read_text())

    if args.model:
        registry = [e for e in registry if e["model_file"] == args.model]
        if not registry:
            print(f"ERROR: '{args.model}' not found in models_registry.json")
            sys.exit(1)

    print(f"\n══ Step 3 / 3 — Evaluate {len(registry)} model(s) on test set ══")
    for entry in registry:
        model_path = MODELS_DIR / entry["model_file"]
        evaluate_model_on_test(model_path, force=args.force)

    print("\n[done] All steps complete.")
    print(f"  Test geo  : {TEST_GEO_PARQUET}")
    print(f"  Metrics   : {OUTPUT_DIR}/test_metrics_*.parquet")


if __name__ == "__main__":
    main()
