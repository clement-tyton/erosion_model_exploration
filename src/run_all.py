"""
Full pipeline: for every model in models_registry.json
  1. Download the tiles JSON manifest from S3  →  data/<dataset_name>.json
  2. Download all NPZ tiles listed in that manifest  →  data/<dataset_name>/
  3. Run inference + compute metrics  →  output/metrics_<model_stem>.parquet

Shared datasets (e.g. v3_ep78 and v3_ep80 both use v3_split_test_final)
are downloaded only once — step 1+2 are skipped when the directory already exists.

Usage:
    python -m src.run_all               # full pipeline, skip cached parquets
    python -m src.run_all --force       # force re-evaluation (tiles not re-downloaded)
    python -m src.run_all --dry-run     # show status, no I/O
"""

import argparse
import json
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

# ROOT = Path("./")

ROOT             = Path(__file__).parent.parent
REGISTRY_PATH    = ROOT / "models_registry.json"
MODELS_DIR       = ROOT / "models"
DATA_ROOT        = ROOT / "data"
OUTPUT_DIR       = ROOT / "output"
TILES_JSON_DIR   = ROOT / "tiles_locations_json"   # manifest JSONs live here

load_dotenv(ROOT / ".env")

# ── S3 client (one per thread) ────────────────────────────────────────────────
_local = threading.local()

def _s3_client():
    if not hasattr(_local, "client"):
        import boto3
        from botocore.config import Config
        _local.client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['AWS_S3_ENDPOINT']}",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
        )
    return _local.client


# ── Path helpers ──────────────────────────────────────────────────────────────
def _paths(entry: dict) -> tuple[Path, Path, Path]:
    """Return (model_path, tiles_json_path, data_dir)."""
    model_path = MODELS_DIR / entry["model_file"]
    tiles_json = TILES_JSON_DIR / f"{entry['dataset_name']}.json"
    data_dir   = DATA_ROOT / entry["dataset_name"]
    return model_path, tiles_json, data_dir


# ── Step 0: download model checkpoint ────────────────────────────────────────
def _download_model(entry: dict, model_path: Path) -> bool:
    s3_uri = entry.get("s3_model", "")
    if not s3_uri.startswith("s3://"):
        print(f"   [ERROR] s3_model not set in registry for {entry['model_file']}")
        return False
    # Parse  s3://bucket/key
    parts  = s3_uri[len("s3://"):].split("/", 1)
    bucket, key = parts[0], parts[1]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"   [model] downloading from {s3_uri}")
    try:
        _s3_client().download_file(bucket, key, str(model_path))
        print(f"   [model] saved → {model_path.name}")
        return True
    except Exception as e:
        print(f"   [ERROR] model download failed: {e}")
        return False


# ── Step 1: download JSON manifest ───────────────────────────────────────────
def _download_manifest(entry: dict, tiles_json: Path) -> bool:
    TILES_JSON_DIR.mkdir(parents=True, exist_ok=True)
    bucket  = os.environ.get("S3_FILE_BUCKET")
    s3_key  = f"{entry['tiles_json_id']}.json"
    print(f"   [manifest] {s3_key} → {tiles_json}")
    try:
        _s3_client().download_file(bucket, s3_key, str(tiles_json))
        return True
    except Exception as e:
        print(f"   [ERROR] manifest download failed: {e}")
        return False


# ── Step 2: download NPZ tiles ────────────────────────────────────────────────
def _download_tiles(entry: dict, tiles_json: Path, data_dir: Path, workers: int = 32) -> bool:
    from src.download_tiles import main as _dl
    prefix = entry.get("s3_tiles_prefix")
    if prefix == "FILL_ME_IN":
        print(f"   [ERROR] s3_tiles_prefix not set in registry for {entry['dataset_name']}")
        print(f"   → Go to TytonAI ObjectTrain workflow, find the training data activity")
        print(f"     output, copy the S3 prefix UUID, and add it to models_registry.json")
        return False
    # prefix=None → v1 flat layout (no prefix); prefix=uuid → standard layout
    print(f"   [tiles] prefix={prefix or '(none — flat)'} → {data_dir}")
    try:
        _dl(tiles_json=tiles_json, data_dir=data_dir, s3_prefix=prefix, workers=workers)
        return True
    except Exception as e:
        print(f"   [ERROR] tile download failed: {e}")
        return False


# ── Step 2.5: build geo parquet ───────────────────────────────────────────────
def _build_geo(entry: dict, tiles_json: Path, data_dir: Path, force: bool) -> bool:
    from src.build_geo import build_geo, geo_parquet_path
    dataset_name = entry["dataset_name"]
    if geo_parquet_path(dataset_name).exists() and not force:
        print(f"   [geo] already built — {geo_parquet_path(dataset_name).name}")
        return True
    print(f"   [geo] building geographic index…")
    try:
        build_geo(dataset_name=dataset_name, tiles_json=tiles_json,
                  data_dir=data_dir, force=force)
        return True
    except Exception as e:
        print(f"   [ERROR] geo build failed: {e}")
        return False


# ── Step 3: evaluate ──────────────────────────────────────────────────────────
def _evaluate(entry: dict, model_path: Path, tiles_json: Path,
              data_dir: Path, force: bool) -> bool:
    from src.evaluate import run_evaluation
    print(f"   [evaluate] running inference…")
    try:
        run_evaluation(
            force=force,
            model_path=model_path,
            tiles_json_path=tiles_json,
            data_dir=data_dir,
        )
        return True
    except Exception as e:
        print(f"   [ERROR] evaluation failed: {e}")
        return False


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_all(force: bool = False, dry_run: bool = False, workers: int = 32) -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    # from pprint import pprint
    # pprint(registry)
    print(f"[run_all] {len(registry)} models in registry\n")

    for entry in registry:
        # entry = registry[0] # --- IGNORE ---   
        model_path, tiles_json, data_dir = _paths(entry)
        stem        = model_path.stem
        out_parquet = OUTPUT_DIR / f"metrics_{stem}.parquet"

        print(f"── {stem}")
        print(f"   dataset : {entry['dataset_name']}  ({entry['tiles_json_id'][:8]}…)")
        print(f"   tiles   : {data_dir}")
        print(f"   parquet : {out_parquet.name}")

        # Model must exist locally — download from S3 if not
        if not model_path.exists():
            if not _download_model(entry, model_path):
                print()
                continue

        if dry_run:
            from src.build_geo import geo_parquet_path
            manifest_ok = tiles_json.exists()
            tiles_ok    = data_dir.exists() and any(data_dir.iterdir()) if data_dir.exists() else False
            geo_ok      = geo_parquet_path(entry["dataset_name"]).exists()
            parquet_ok  = out_parquet.exists()
            _prefix = entry.get("s3_tiles_prefix")
            _prefix_ok = _prefix != "FILL_ME_IN"  # None (flat) or real UUID are both valid
            _prefix_label = ("(flat — no prefix)" if _prefix is None
                             else (_prefix[:8] + "…" if _prefix_ok else "❌ MISSING — fill in models_registry.json"))
            print(f"   manifest : {'✅' if manifest_ok else '❌ will download'}")
            print(f"   s3_prefix: {'✅ ' + _prefix_label if _prefix_ok else _prefix_label}")
            print(f"   tiles    : {'✅' if tiles_ok    else '❌ will download'}")
            print(f"   geo      : {'✅' if geo_ok      else '❌ will build'}")
            print(f"   parquet  : {'✅ cached' if parquet_ok and not force else ('🔄 will re-run' if parquet_ok and force else '❌ will evaluate')}")
            print()
            continue

        # ── Step 1: manifest ──────────────────────────────────────────────────
        if not tiles_json.exists():
            if not _download_manifest(entry, tiles_json):
                print()
                continue

        # ── Step 2: tiles ─────────────────────────────────────────────────────
        if not data_dir.exists() or not any(data_dir.iterdir()):
            if not _download_tiles(entry, tiles_json, data_dir, workers=workers): # workers = 32
                print()
                continue
        else:
            n = sum(1 for _ in data_dir.iterdir())
            print(f"   [tiles] already present ({n} files)")

        # ── Step 2.5: geo index ───────────────────────────────────────────────
        _build_geo(entry, tiles_json, data_dir, force=force)

        # ── Step 3: evaluate ──────────────────────────────────────────────────
        if out_parquet.exists() and not force:
            print(f"   [parquet] cached — use --force to re-evaluate")
            print()
            continue

        _evaluate(entry, model_path, tiles_json, data_dir, force=force)
        print()

    print("[run_all] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full pipeline: download + evaluate all models")
    parser.add_argument("--force",   action="store_true",
                        help="Re-run evaluation even if parquet exists (tiles are NOT re-downloaded)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show status of each step without doing anything")
    parser.add_argument("--workers", type=int, default=32,
                        help="Parallel workers for tile download (default: 32)")
    args = parser.parse_args()
    run_all(force=args.force, dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
