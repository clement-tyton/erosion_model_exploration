"""
Download all tile NPZ files listed in the balanced tiles JSON.

Usage:
    python src/download_tiles.py [--workers 12]

Uses boto3 with a thread pool (one client per thread).
Skips files that already exist locally. Safe to re-run (resume).
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

ROOT = Path(__file__).parent.parent

# Defaults (legacy — ef1410ef dataset already in data/training_data)
_DEFAULT_TILES_JSON = ROOT / "ef1410ef-59ab-4821-b044-11f8ef6a040a.json"
_DEFAULT_DATA_DIR   = ROOT / "data" / "training_data"

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

# One boto3 client per thread
_local = threading.local()
_transfer_config = TransferConfig(use_threads=False)  # we handle threading ourselves


def get_client(workers: int):
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


def download_one(filename: str, s3_key: str, data_dir: Path, workers: int) -> tuple[str, str]:
    dest = data_dir / filename
    if dest.exists():
        return filename, "skipped"
    try:
        get_client(workers).download_file(
            BUCKET, s3_key, str(dest), Config=_transfer_config
        )
        return filename, "ok"
    except Exception as e:
        return filename, str(e)


def main(
    workers: int = 32,
    tiles_json: Path = _DEFAULT_TILES_JSON,
    data_dir: Path = _DEFAULT_DATA_DIR,
    s3_prefix: str | None = None,
):
    data_dir = Path(data_dir)
    tiles_json = Path(tiles_json)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(tiles_json) as f:
        tiles = json.load(f)["balanced_tiles"]

    all_files = []
    for tile in tiles:
        all_files.append(tile["imagery_file"])
        all_files.append(tile["mask_file"])
    all_files = list(dict.fromkeys(all_files))

    to_download = [f for f in all_files if not (data_dir / f).exists()]
    skipped     = len(all_files) - len(to_download)

    # Build S3 key: prefix/filename  or  just filename if no prefix
    def _s3_key(filename: str) -> str:
        return f"{s3_prefix}/{filename}" if s3_prefix else filename

    print(f"S3 prefix: {s3_prefix or '(none — bare filename)'}")
    print(f"Total  : {len(all_files)} files")
    print(f"Present: {skipped} (skipping)")
    print(f"To get : {len(to_download)} — {workers} workers\n")

    if not to_download:
        print("Nothing to download.")
        return

    failed = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, f, _s3_key(f), data_dir, workers): f
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
    parser = argparse.ArgumentParser(
        description="Download tile NPZ files listed in a balanced tiles JSON."
    )
    parser.add_argument("--tiles-json", type=str, default=None,
                        help=f"Path to tiles JSON manifest (default: {_DEFAULT_TILES_JSON.name})")
    parser.add_argument("--data-dir", type=str, default=None,
                        help=f"Destination directory for NPZ files (default: {_DEFAULT_DATA_DIR})")
    parser.add_argument("--s3-prefix", type=str, default=None,
                        help="S3 key prefix where tiles are stored (e.g. f9479ddf-...)")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel download workers (default: 32)")
    args = parser.parse_args()
    main(
        workers=args.workers,
        tiles_json=Path(args.tiles_json) if args.tiles_json else _DEFAULT_TILES_JSON,
        data_dir=Path(args.data_dir)    if args.data_dir   else _DEFAULT_DATA_DIR,
        s3_prefix=args.s3_prefix,
    )

