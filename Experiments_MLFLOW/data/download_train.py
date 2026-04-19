"""
Download train NPZ tiles for Experiments_MLFLOW.

Two-phase process:
  1. Download the balanced tiles manifest JSON from S3:
       key = balanced_tiles_path from DATABALANCE_CONFIG.JSON
       saved to Experiments_MLFLOW/metadata/balanced_tiles.json
  2. Download every imagery_file + mask_file listed in that manifest.
       Output: Experiments_MLFLOW/data/train_data/

S3 layout: no prefix — bare filename at bucket root.

Usage:
    python -m Experiments_MLFLOW.data.download_train [--workers 32]
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from dotenv import load_dotenv
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent.parent
ROOT = EXPERIMENTS_DIR.parent
METADATA_DIR = EXPERIMENTS_DIR / "metadata"
DEFAULT_DATA_DIR = EXPERIMENTS_DIR / "data" / "train_data"
DATABALANCE_CONFIG = METADATA_DIR / "DATABALANCE_CONFIG.JSON"
BALANCED_TILES_DEST = METADATA_DIR / "balanced_tiles.json"

# ── S3 config (from project root .env) ────────────────────────────────────────
load_dotenv(ROOT / ".env")

BUCKET        = os.environ.get("S3_FILE_BUCKET")
ENDPOINT      = os.environ.get("AWS_S3_ENDPOINT")
REGION        = os.environ.get("AWS_REGION", "us-east-1")
ACCESS_KEY    = os.environ.get("AWS_ACCESS_KEY_ID")
SECRET_KEY    = os.environ.get("AWS_SECRET_ACCESS_KEY")
SESSION_TOKEN = os.environ.get("AWS_SESSION_TOKEN")

if not BUCKET or not ENDPOINT:
    print("ERROR: S3_FILE_BUCKET / AWS_S3_ENDPOINT not set in .env")
    sys.exit(1)

# ── Thread-local S3 client (one per thread) ───────────────────────────────────
_local = threading.local()
_transfer_config = TransferConfig(use_threads=False)


def get_client(workers: int = 32):
    if not hasattr(_local, "client"):
        _local.client = boto3.client(
            "s3",
            endpoint_url=f"https://{ENDPOINT}",
            region_name=REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            aws_session_token=SESSION_TOKEN,
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=workers,
            ),
        )
    return _local.client


def download_one(filename: str, data_dir: Path, workers: int = 32) -> tuple[str, str]:
    """Download a single NPZ file (bare key = filename). Skips if already present."""
    dest = data_dir / filename
    if dest.exists():
        return filename, "skipped"
    try:
        get_client(workers).download_file(
            BUCKET, filename, str(dest), Config=_transfer_config
        )
        return filename, "ok"
    except Exception as e:
        return filename, str(e)


def download_manifest(balanced_tiles_path: str) -> None:
    """Download the balanced tiles JSON manifest from S3 and save locally."""
    if BALANCED_TILES_DEST.exists():
        print(f"Manifest already present: {BALANCED_TILES_DEST}")
        return
    print(f"Downloading manifest: {balanced_tiles_path} → {BALANCED_TILES_DEST}")
    client = get_client()
    client.download_file(BUCKET, balanced_tiles_path, str(BALANCED_TILES_DEST))
    print("Manifest downloaded.")


def main(workers: int = 32, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: get the manifest ───────────────────────────────────────────────
    with open(DATABALANCE_CONFIG) as f:
        db_config = json.load(f)
    balanced_tiles_path: str = db_config["balanced_tiles_path"]
    download_manifest(balanced_tiles_path)

    # ── Step 2: collect unique file names ─────────────────────────────────────
    with open(BALANCED_TILES_DEST) as f:
        tiles = json.load(f)["balanced_tiles"]

    all_files: list[str] = []
    for tile in tiles:
        all_files.append(tile["imagery_file"])
        all_files.append(tile["mask_file"])
    all_files = list(dict.fromkeys(all_files))  # deduplicate, preserve order

    to_download = [f for f in all_files if not (data_dir / f).exists()]
    n_skip = len(all_files) - len(to_download)

    print(f"Total unique files : {len(all_files)}")
    print(f"Already on disk    : {n_skip} (skipping)")
    print(f"To download        : {len(to_download)} — {workers} workers\n")

    if not to_download:
        print("Nothing to download.")
        return

    failed: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, f, data_dir, workers): f
            for f in to_download
        }
        with tqdm(total=len(to_download), unit="file", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                _, status = future.result()
                if status not in ("ok", "skipped"):
                    failed.append((futures[future], status))
                    bar.set_postfix(failed=len(failed), refresh=False)
                bar.update(1)

    print(f"\nDone. Failed: {len(failed)}")
    if failed:
        print("\nFailed files:")
        for fname, err in failed:
            print(f"  {fname}\n    {err}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download train NPZ tiles from S3.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    main(workers=args.workers, data_dir=Path(args.data_dir))
