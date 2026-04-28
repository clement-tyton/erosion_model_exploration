"""
One-time setup: register the 3 MLflow-trained models into the dashboard.

What this script does:
  1. Creates symlinks in models/ pointing to the MLflow checkpoint files
     (avoids copying ~500 MB per model).
  2. Appends entries to models_registry.json for each model (idempotent —
     skips entries that are already present).

Run from the repo root:
    python Experiments_MLFLOW/register_mlflow_models.py
"""

import json
import os
from pathlib import Path

ROOT            = Path(__file__).parent.parent
EXPERIMENTS_DIR = Path(__file__).parent
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
MODELS_DIR      = ROOT / "models"
REGISTRY_PATH   = ROOT / "models_registry.json"

# ── Model definitions ──────────────────────────────────────────────────────────

MLFLOW_MODELS = [
    {
        "run_name":    "segf",
        "model_file":  "mlflow_segf_epoch200.pth",
        "version":     "mlflow_segf",
        "epoch":       200,
        "description": "MLflow SegFormer (res2net encoder) — epoch 200",
        "dataset_name": "mlflow_balanced_tiles",
        "tiles_json_id": "ef1410ef-59ab-4821-b044-11f8ef6a040a",
    },
    {
        "run_name":    "unet_baseline",
        "model_file":  "mlflow_unet_baseline_epoch200.pth",
        "version":     "mlflow_unet",
        "epoch":       200,
        "description": "MLflow UNet baseline (res2net encoder) — epoch 200",
        "dataset_name": "mlflow_balanced_tiles",
        "tiles_json_id": "ef1410ef-59ab-4821-b044-11f8ef6a040a",
    },
    {
        "run_name":    "segformer_mit_b3_stable",
        "model_file":  "mlflow_segformer_mit_b3_stable_epoch200.pth",
        "version":     "mlflow_segformer_mit_b3",
        "epoch":       200,
        "description": "MLflow SegFormer MiT-B3 — epoch 200",
        "dataset_name": "mlflow_balanced_tiles",
        "tiles_json_id": "ef1410ef-59ab-4821-b044-11f8ef6a040a",
    },
]

# ── Step 1: Create symlinks ────────────────────────────────────────────────────

MODELS_DIR.mkdir(exist_ok=True)

print("── Creating symlinks in models/ ──")
for m in MLFLOW_MODELS:
    src = CHECKPOINTS_DIR / m["run_name"] / "epoch_0200.pth"
    dst = MODELS_DIR / m["model_file"]

    if not src.exists():
        print(f"  SKIP  {m['model_file']} — source not found: {src}")
        continue

    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            print(f"  OK    {m['model_file']} (symlink already correct)")
        else:
            print(f"  WARN  {m['model_file']} already exists but points elsewhere — skipping")
        continue

    dst.symlink_to(src)
    print(f"  LINK  {m['model_file']} → {src.relative_to(ROOT)}")

# ── Step 2: Update models_registry.json ───────────────────────────────────────

print("\n── Updating models_registry.json ──")
with open(REGISTRY_PATH) as f:
    registry: list[dict] = json.load(f)

existing_files = {entry["model_file"] for entry in registry}
added = 0

for m in MLFLOW_MODELS:
    if m["model_file"] in existing_files:
        print(f"  SKIP  {m['model_file']} (already in registry)")
        continue

    entry = {k: v for k, v in m.items() if k != "run_name"}
    registry.append(entry)
    print(f"  ADD   {m['model_file']}")
    added += 1

if added:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"\nRegistry updated — {added} entries added.")
else:
    print("\nRegistry unchanged — all entries already present.")

# ── Step 3: Build geographic index ────────────────────────────────────────────

print("\n── Building geographic index for MLflow train tiles ──")
_train_data_dir = EXPERIMENTS_DIR / "data" / "train_data"
_tiles_json     = EXPERIMENTS_DIR / "metadata" / "balanced_tiles.json"
_geo_out        = ROOT / "output" / "tiles_geo_mlflow_balanced_tiles.parquet"

if not _train_data_dir.exists() or not any(_train_data_dir.glob("*.npz")):
    print(f"  SKIP  No NPZ files found in {_train_data_dir}")
    print("        Run train evaluation first to download tiles, then re-run this script.")
elif _geo_out.exists():
    print(f"  OK    {_geo_out.name} already exists (use --force-geo to rebuild)")
    import sys
    if "--force-geo" not in sys.argv:
        pass
    else:
        from src.build_geo import build_geo
        build_geo(
            dataset_name="mlflow_balanced_tiles",
            tiles_json=_tiles_json,
            data_dir=_train_data_dir,
            workers=8,
            force=True,
        )
else:
    try:
        from src.build_geo import build_geo
        build_geo(
            dataset_name="mlflow_balanced_tiles",
            tiles_json=_tiles_json,
            data_dir=_train_data_dir,
            workers=8,
            force=False,
        )
    except Exception as e:
        print(f"  ERROR building geo index: {e}")
        print("  Run manually: python -m src.build_geo "
              "--dataset-name mlflow_balanced_tiles "
              f"--tiles-json {_tiles_json} "
              f"--data-dir {_train_data_dir}")

# ── Summary ────────────────────────────────────────────────────────────────────

print("\n── Next steps ──")
print("Train metrics (run from repo root, requires Experiments_MLFLOW/data/train_data/):")
for m in MLFLOW_MODELS:
    print(f"""
  python -m src.evaluate --force \\
    --model-path models/{m["model_file"]} \\
    --tiles-json Experiments_MLFLOW/metadata/balanced_tiles.json \\
    --data-dir Experiments_MLFLOW/data/train_data""")

print("\nTest metrics (run from repo root, requires test tiles in Experiments_MLFLOW/data/test_data/):")
for m in MLFLOW_MODELS:
    print(f"""
  python -m DASHBOARD_FROM_TYTONAI.evaluate_test --force --skip-download --skip-geo \\
    --model {m["model_file"]}""")

print("\nThen launch: streamlit run DASHBOARD_FROM_TYTONAI/app.py")
