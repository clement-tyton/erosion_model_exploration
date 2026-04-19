"""
Download test NPZ tiles for Experiments_MLFLOW.

Reads the local EROSION_DATASET_TEST_METADATA.json (flat list — no balanced_tiles wrapper)
and downloads every imagery_file + mask_file entry from S3.

S3 layout: no prefix — bare filename at bucket root.

Usage:
    python -m Experiments_MLFLOW.data.download_test [--workers 32]
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
TEST_METADATA_JSON = METADATA_DIR / "EROSION_DATASET_TEST_METADATA.json"
DEFAULT_DATA_DIR = EXPERIMENTS_DIR / "data" / "test_data"

# ── S3 config ─────────────────────────────────────────────────────────────────
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
    """Download a single NPZ file (bare key). Skips if already present."""
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


def main(workers: int = 32, data_dir: Path = DEFAULT_DATA_DIR) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(TEST_METADATA_JSON) as f:
        entries = json.load(f)

    all_files: list[str] = []
    for entry in entries:
        all_files.append(entry["imagery_file"])
        all_files.append(entry["mask_file"])
    all_files = list(dict.fromkeys(all_files))

    to_download = [f for f in all_files if not (data_dir / f).exists()]
    n_skip = len(all_files) - len(to_download)

    print(f"Test entries       : {len(entries)}")
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
    parser = argparse.ArgumentParser(description="Download test NPZ tiles from S3.")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    main(workers=args.workers, data_dir=Path(args.data_dir))
