"""
Central configuration for the erosion evaluation pipeline.
All values derived from files_from_tytonai/metadata_model_train.json.
"""

from pathlib import Path

# ── Project root (parent of src/) ───────────────────────────────────────────
ROOT = Path(__file__).parent.parent

# ── Input data ───────────────────────────────────────────────────────────────
TILES_JSON = ROOT / "ef1410ef-59ab-4821-b044-11f8ef6a040a.json"
DATA_DIR = ROOT / "data" / "training_data"   # NPZ files live here after unzip
MODEL_PATH = ROOT / "models" / "model_epoch_80.pth"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / "output"
METRICS_CSV = OUTPUT_DIR / "metrics.csv"
METRICS_PARQUET = OUTPUT_DIR / "metrics.parquet"
TILES_DIR = OUTPUT_DIR / "tiles"   # PNG visualisations

# ── Model architecture ────────────────────────────────────────────────────────
ENCODER = "timm-res2net101_26w_4s"
ENCODER_DEPTH = 5
ACTIVATION = "softmax2d"
MODEL_TYPE = "Unet"

# ── Bands ─────────────────────────────────────────────────────────────────────
# Bands the model was trained on (order matters — matches train_mean / train_std)
MODEL_BANDS = ["RED", "GREEN", "BLUE", "DSM_NORMALIZED"]

# ── Normalisation (from training metadata) ────────────────────────────────────
# Order: RED, GREEN, BLUE, DSM_NORMALIZED
TRAIN_MEAN = [150.73301134918557, 123.75755228360018, 92.57823716578613, -9.734063808604613]
TRAIN_STD = [39.721974708734216, 34.06117915518031, 30.092062243775406, 4.684211737168346]

# ── Classes ───────────────────────────────────────────────────────────────────
# Raw pixel labels in mask NPZ files: 0 = background, 1 = no-erosion, 14 = erosion
# Model output channels:             channel 0 = class 1,             channel 1 = class 14
CLASS_LIST = [1, 14]
IGNORE_INDEX = 255   # background pixels are remapped to this and excluded from metrics

# Mapping: raw mask pixel value → model output channel index
# Background (0) → IGNORE_INDEX so it is excluded from metrics
LABEL_TO_CHANNEL = {0: IGNORE_INDEX, 1: 0, 14: 1}

# ── DataLoader ────────────────────────────────────────────────────────────────
BATCH_SIZE = 8
NUM_WORKERS = 4
