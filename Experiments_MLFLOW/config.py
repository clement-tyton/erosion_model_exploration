"""
Central configuration for Experiments_MLFLOW.

Values loaded from metadata JSON files; a few constants are fixed overrides.
"""

import json
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
METADATA_DIR    = EXPERIMENTS_DIR / "metadata"

# ── Load from DATABALANCE_CONFIG.JSON ─────────────────────────────────────────
with open(METADATA_DIR / "DATABALANCE_CONFIG.JSON") as _f:
    _db = json.load(_f)

TRAIN_MEAN: list[float] = _db["img_mean"]          # [RED, GREEN, BLUE, DSM_NORMALIZED]
TRAIN_STD:  list[float] = _db["img_std"]
CLASS_WEIGHTS: list[float] = _db["class_weights"]  # [1.0, 1.764] — weight for CE loss

# ── Load from OBJECT_TRAIN_INPUT_JSON ─────────────────────────────────────────
with open(METADATA_DIR / "OBJECT_TRAIN_INPUT_JSON") as _f:
    _ot = json.load(_f)

INITIAL_LR:     float = float(_ot["initial_lr"])             # 0.001
LR_STEP_SIZE:   int   = int(_ot["lr_schedular_step_size"])   # 133 (epochs)
LR_DECAY:       float = float(_ot["lr_schedular_dec_rate"])  # 0.5
NUM_EPOCHS:     int   = int(_ot["num_epochs"])                # 400

# ── Fixed overrides ────────────────────────────────────────────────────────────
BATCH_SIZE:       int       = 32                                        # physical batch (×accum → effective)
ACCUMULATION_STEPS: int     = 8                                         # effective batch = 32 × 8 = 256
MODEL_BANDS: list[str]      = ["RED", "GREEN", "BLUE", "DSM_NORMALIZED"]
IGNORE_INDEX:     int       = 255
CLASS_LIST:       list[int] = [1, 14]
LABEL_TO_CHANNEL: dict      = {0: IGNORE_INDEX, 1: 0, 14: 1}
NUM_WORKERS:      int       = 4
NUM_CLASSES:      int       = 2
IN_CHANNELS:      int       = 4
ENCODER_NAME:     str       = "timm-res2net101_26w_4s"   # UNet default encoder (for MLflow logging)

# ── Paths ─────────────────────────────────────────────────────────────────────
BALANCED_TILES_JSON  = METADATA_DIR / "balanced_tiles.json"
TEST_METADATA_JSON   = METADATA_DIR / "EROSION_DATASET_TEST_METADATA.json"
TRAIN_DATA_DIR       = EXPERIMENTS_DIR / "data" / "train_data"
TEST_DATA_DIR        = EXPERIMENTS_DIR / "data" / "test_data"
CHECKPOINTS_DIR      = EXPERIMENTS_DIR / "checkpoints"
RESULTS_DIR          = EXPERIMENTS_DIR / "results"
