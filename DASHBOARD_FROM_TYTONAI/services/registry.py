"""
Registry & path-resolution helpers.

Exports
-------
ROOT, MODELS_DIR, _TILES_JSON_DIR, _TEST_DATA_DIR, _TEST_METADATA_JSON
_MODEL_DISPLAY_ORDER, _REGISTRY_PATH, _REGISTRY
_model_sort_key, _registry_entry
_metrics_path, _test_metrics_path, _test_geo_path
_model_tiles_json, _model_data_dir
"""
from __future__ import annotations

import json as _json_mod
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# services/registry.py → services/ → DASHBOARD_FROM_TYTONAI/ → project root
ROOT = Path(__file__).parent.parent.parent

MODELS_DIR        = ROOT / "models"
_TILES_JSON_DIR   = ROOT / "tiles_locations_json"
_TEST_DATA_DIR    = ROOT / "Experiments_MLFLOW" / "data" / "test_data"
_TEST_METADATA_JSON = ROOT / "Experiments_MLFLOW" / "metadata" / "EROSION_DATASET_TEST_METADATA.json"

# ── Model display order ───────────────────────────────────────────────────────
_MODEL_DISPLAY_ORDER = [
    "model_v3_split_test_epoch400.pth",
    "model_v3_split_test_epoch399.pth",
    "model_v3_split_test_epoch_243.pth",
    "model_v3_split_test_epoch_95.pth",
    "model_v3_split_test_epoch80.pth",
    "model_v3_split_test_epoch78.pth",
    "model_v3_split_test_epoch50.pth",
    "model_v2_no_erosion_td_epoch50.pth",
    "model_v1_jaswinder_epoch50.pth",
    "model_finetuned_tytonai_epoch5.pth",
]


def _model_sort_key(p: Path) -> int:
    try:
        return _MODEL_DISPLAY_ORDER.index(p.name)
    except ValueError:
        return len(_MODEL_DISPLAY_ORDER)


# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY_PATH: Path = ROOT / "models_registry.json"
_REGISTRY: dict[str, dict] = {}
if _REGISTRY_PATH.exists():
    for _e in _json_mod.loads(_REGISTRY_PATH.read_text()):
        _REGISTRY[_e["model_file"]] = _e


def _registry_entry(model_name: str) -> dict:
    return _REGISTRY.get(model_name, {})


# ── Metrics paths ─────────────────────────────────────────────────────────────
def _metrics_path(stem: str) -> Path | None:
    from src.config import OUTPUT_DIR
    for p in [
        OUTPUT_DIR / f"metrics_{stem}.parquet",
        OUTPUT_DIR / f"metrics_{stem}.csv",
        OUTPUT_DIR / "metrics.parquet",
        OUTPUT_DIR / "metrics.csv",
    ]:
        if p.exists():
            return p
    return None


def _test_metrics_path(stem: str) -> Path | None:
    from src.config import OUTPUT_DIR
    p = OUTPUT_DIR / f"test_metrics_{stem}.parquet"
    return p if p.exists() else None


def _test_geo_path() -> Path:
    from src.config import OUTPUT_DIR
    return OUTPUT_DIR / "tiles_geo_test.parquet"


# ── Tile JSON & data dir ──────────────────────────────────────────────────────
_TILES_JSON_SEARCH_DIRS = [
    ROOT,
    _TILES_JSON_DIR,
    ROOT / "Experiments_MLFLOW" / "metadata",
]


def _model_tiles_json(model_name: str) -> Path:
    entry = _registry_entry(model_name)
    for key in ("dataset_name", "tiles_json_id"):
        if not (entry and key in entry):
            continue
        for search_dir in _TILES_JSON_SEARCH_DIRS:
            p = search_dir / f"{entry[key]}.json"
            if p.exists():
                return p
    from src.config import TILES_JSON
    return TILES_JSON


def _model_color(model_name: str) -> str:
    """Return a hex color for a model based on its version group."""
    entry = _registry_entry(model_name)
    version = entry.get("version", "")
    if version.startswith("tytonai"):
        return "#1ABC9C"       # teal   — TytonAI production
    if version.startswith("mlflow"):
        if "mit_b3" in version:
            return "#F39C12"   # amber  — SegFormer MiT-B3
        if "segf" in version:
            return "#E67E22"   # orange — SegFormer res2net
        return "#D35400"       # dark orange — UNet baseline / other mlflow
    if version.startswith("v4"):
        return "#9B59B6"       # purple — finetuned
    if version.startswith("v3"):
        return "#2ECC71"       # green  — v3
    if version.startswith("v2"):
        return "#3498DB"       # blue   — v2
    return "#95A5A6"           # grey   — v1 / unknown


_MLFLOW_TRAIN_DATA_DIR = ROOT / "Experiments_MLFLOW" / "data" / "train_data"


def _model_data_dir(model_name: str) -> Path:
    entry = _registry_entry(model_name)
    if entry and "dataset_name" in entry:
        if entry["dataset_name"] == "mlflow_balanced_tiles":
            return _MLFLOW_TRAIN_DATA_DIR
        p = ROOT / "data" / entry["dataset_name"]
        if p.exists():
            return p
    from src.config import DATA_DIR
    return DATA_DIR
