# Erosion R&D — Evaluation & Exploration Dashboard

Internal tool for evaluating and comparing erosion detection models trained on drone imagery (Western Australia).

---

## What this does

A UNet segmentation model classifies each pixel of a multispectral tile as **erosion** (class 14) or **no-erosion** (class 1).  
This repo provides:
- Batch inference on all training tiles
- Per-tile metrics (F1, precision, recall, IOU) stored in Parquet
- A Streamlit dashboard to explore results, compare models, and visualise individual tiles on a map

---

## Project structure

```
erosion_R_and_D/
│
├── models_registry.json             # Source of truth: model → dataset mapping (see below)
│
├── models/                          # Model checkpoints (.pth) — downloaded by run_all
│   ├── model_v1_jaswinder_epoch50.pth
│   ├── model_v2_no_erosion_td_epoch50.pth
│   ├── model_v3_split_test_epoch50.pth
│   ├── model_v3_split_test_epoch78.pth
│   ├── model_v3_split_test_epoch80.pth
│   └── model_finetuned_tytonai_epoch5.pth
│
├── tiles_locations_json/            # Tile manifest JSONs — downloaded by run_all
│   ├── v1_jaswinder.json
│   ├── v2_no_erosion_td.json
│   ├── v3_split_test_ep50.json
│   ├── v3_split_test_final.json
│   └── finetuned_tytonai_ep5.json
│
├── data/                            # Downloaded tile NPZ files — one dir per dataset
│   ├── v1_jaswinder/
│   ├── v2_no_erosion_td/
│   ├── v3_split_test_ep50/
│   ├── v3_split_test_final/         # shared by ep78 and ep80
│   └── finetuned_tytonai_ep5/
│
├── output/                          # Generated outputs (all skipped if already present)
│   ├── metrics_<model_stem>.parquet # Per-tile metrics (one file per model)
│   ├── metrics_<model_stem>.csv     # Same data as parquet
│   ├── tiles_geo_<dataset>.parquet  # WGS84 centroids (one file per unique dataset)
│   └── tiles/                       # PNG visualisations (one subdirectory per model)
│
├── .env                             # AWS credentials (not committed)
│
└── src/
    ├── config.py                    # Paths, model hyperparameters, normalisation stats
    ├── dataset.py                   # NPZ loading, band selection, normalisation, DataLoader
    ├── model.py                     # SMP UNet builder + checkpoint loading
    ├── evaluate.py                  # Batch inference → per-tile metrics → Parquet
    ├── visualize.py                 # PNG generation (side-by-side masks + contour overlay)
    ├── build_geo.py                 # Extract WGS84 centroids from NPZ tiles (per dataset)
    ├── download_tiles.py            # Download NPZ files from S3 given a manifest JSON
    ├── run_all.py                   # Full pipeline: download → evaluate for all models
    └── app.py                       # Streamlit dashboard
```

---

## Model registry

`models_registry.json` is the single source of truth linking each model to its training dataset.  
`run_all.py` reads this file and drives every step of the pipeline.

```json
{
  "model_file":    "model_v3_split_test_epoch80.pth",
  "version":       "v3",
  "epoch":         80,
  "description":   "v2 + no-erosion TD + split test — epoch 80",
  "dataset_name":  "v3_split_test_final",
  "tiles_json_id": "ef1410ef-59ab-4821-b044-11f8ef6a040a",
  "s3_model":      "s3://bucket/path/model_epoch_80.pth"
}
```

### How each field is used

| Field | Used by | Purpose |
|---|---|---|
| `model_file` | run_all, app | Local filename under `models/`; also drives the output parquet name (`metrics_<stem>.parquet`) |
| `version`, `epoch`, `description` | app | Display labels in the dashboard Compare tab |
| `dataset_name` | run_all, app, build_geo | Local tile directory (`data/<dataset_name>/`) and manifest path (`tiles_locations_json/<dataset_name>.json`). Models sharing the same value **reuse the same downloaded tiles and geo index** — no redundant downloads |
| `tiles_json_id` | run_all | S3 UUID used to download the tile manifest: `<bucket>/<tiles_json_id>.json` → `tiles_locations_json/<dataset_name>.json` |
| `s3_model` | run_all | Full S3 URI for the model checkpoint — downloaded once to `models/<model_file>` |

### All registered models

| model_file | version | epoch | dataset_name | Description |
|---|---|---|---|---|
| model_v1_jaswinder_epoch50.pth | v1 | 50 | v1_jaswinder | Original Jaswinder baseline |
| model_v2_no_erosion_td_epoch50.pth | v2 | 50 | v2_no_erosion_td | v1 + added no-erosion training data |
| model_v3_split_test_epoch50.pth | v3 | 50 | v3_split_test_ep50 | v2 + train/test split, early checkpoint |
| model_v3_split_test_epoch78.pth | v3 | 78 | v3_split_test_final | v2 + split test — epoch 78 |
| model_v3_split_test_epoch80.pth | v3 | 80 | v3_split_test_final | v2 + split test — best checkpoint |
| model_finetuned_tytonai_epoch5.pth | v4_finetuned | 5 | finetuned_tytonai_ep5 | v3 finetuned on TytonAI data — epoch 5 |

> `v3_split_test_final` is shared by ep78 and ep80: tiles and geo index are downloaded once, metrics parquets are separate.

---

## Skip / cache logic

Every step is idempotent — re-running `run_all` never re-downloads or re-computes anything unless it is genuinely missing or `--force` is passed.

| Step | Skipped when |
|---|---|
| Model download | `models/<model_file>` exists |
| Manifest download | `tiles_locations_json/<dataset_name>.json` exists |
| Tile download (dataset level) | `data/<dataset_name>/` is non-empty |
| Individual tile download | `data/<dataset_name>/<file>.npz` already exists (checked in `download_tiles.py`) |
| Geo index build | `output/tiles_geo_<dataset_name>.parquet` exists (unless `--force`) |
| Evaluation | `output/metrics_<model_stem>.parquet` exists (unless `--force`) |
| PNG generation | `output/tiles/<model>/<tile>.png` exists (checked in `visualize.py` and `app.py`) |

`--force` re-runs evaluation and geo build only; **tiles are never re-downloaded by force**.

---

## Quick start

### 1. Install
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

### 2. Set up credentials
```bash
cp .env.example .env   # fill in AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
                       # AWS_SESSION_TOKEN, S3_FILE_BUCKET, AWS_S3_ENDPOINT
```

### 3. Run the full pipeline (all models)
```bash
python -m src.run_all
```

For each model in the registry, in order:
1. Download model checkpoint from S3 (if not present)
2. Download tile manifest JSON from S3 (if not present)
3. Download all NPZ tiles into `data/<dataset_name>/` — 32 parallel workers, per-file skip (if not present)
4. Build geographic index → `output/tiles_geo_<dataset_name>.parquet` (if not present)
5. Run inference → `output/metrics_<model_stem>.parquet` (if not present)

```bash
python -m src.run_all --dry-run    # preview what would run without doing anything
python -m src.run_all --force      # force re-evaluation (tiles are NOT re-downloaded)
python -m src.run_all --workers 64 # change tile download parallelism (default: 32)
```

### 4. Launch the dashboard
```bash
streamlit run src/app.py
```

---

## Dashboard tabs

### Overview
Global KPIs for the selected model:
- **Mean F1 erosion** — macro: average of per-tile F1 scores on erosion tiles
- **Global F1 erosion** — micro: 2·ΣTP / (2·ΣTP + ΣFP + ΣFN) across all pixels
- Precision / Recall / F1 for no-erosion
- A slider filters the KPIs and charts to tiles with ≥ N erosion pixels (useful to exclude very sparse tiles)
- Precision vs Recall scatter (one dot per tile, size = erosion pixel count)
- F1 / Precision / Recall / IOU histograms

### Tile explorer
- Filter by F1 range, minimum erosion pixels
- Sort by any metric or compound preset (e.g. "worst F1 + most erosion pixels")
- Click any row to generate and view the tile visualisation on the fly
- Two visualisation styles:
  - **Side-by-side**: RGB · DSM · predicted mask · true mask
  - **Contour overlay**: red contour = ground truth, blue contour = prediction, overlaid on RGB and DSM

### Map
- All tiles plotted on a dark basemap (carto-darkmatter)
- Colour by recall, F1, precision, or pixel size
- Filter by F1 range and erosion pixel count
- Click a tile dot to load its visualisation inline

### Compare models
- Loads all available `output/metrics_*.parquet` files
- Summary leaderboard: F1, global F1, precision, recall, IOU, dataset, tile count, erosion prevalence
- Bar chart comparing any metric across models
- Scatter: erosion prevalence in dataset vs model F1 (shows data impact on performance)
- Pairwise deep-dive and performance by erosion density

### Raw data
- Full parquet viewer (first 5 000 rows)
- DuckDB SQL console — query the `metrics` view directly

---

## Key design decisions

**Why Parquet + DuckDB?**  
19 752 tiles per model × 6 models. Parquet gives fast columnar reads; DuckDB queries it in-process with SQL — no server, no loading the whole file into RAM.

**Why inference on CPU in the app?**  
CUDA causes segfaults in Streamlit's forked worker process. CPU is slower but stable for on-the-fly single-tile inference. Batch evaluation (`run_all`) uses CUDA when available.

**Why Pad32?**  
The SMP encoder requires spatial dimensions to be multiples of 32. Some tiles are smaller (e.g. 285×122). The app pads before inference and crops the prediction back to the original size.

**Macro vs micro F1**  
- Macro F1 = average of per-tile F1 scores → sensitive to hard tiles regardless of size  
- Micro F1 = pixel-level aggregate across all tiles → weighted by tile size  
Both are shown. Macro is more intuitive for "how often does the model fail completely on a tile"; micro is the true pixel-level accuracy.

**Inference is run on the training dataset, not a held-out test set**  
v3 models have a train/test split but we currently evaluate on all balanced tiles (which includes training tiles). A future step is to download the dedicated test set (Dampier, Dugong, Redtingle) for unbiased evaluation.

---

## Tile format (NPZ)

Each imagery tile is a `.npz` archive with keys:
- `RED`, `GREEN`, `BLUE`, `DSM`, `DSM_NORMALIZED`, `MEP` — band arrays (H × W float32)
- `SRID` — EPSG code (20350 = GDA94 / MGA zone 50, Western Australia)
- `GEO_TRANSFORM` — 9-element row-major 3×3 affine matrix:  
  `[scale_x, shear_x, x_origin, shear_y, scale_y, y_origin, 0, 0, 1]`
- `VERSION` — format version

Mask tiles use the same format with a single band containing pixel labels:
- `0` → background (ignored in metrics)
- `1` → no-erosion
- `14` → erosion

The model uses 4 bands: `RED`, `GREEN`, `BLUE`, `DSM_NORMALIZED`. Tiles missing any of these are filtered out before inference.

---

## Normalisation

The model was trained with per-band mean/std normalisation (values in 0–255 scale):

| Band | Mean | Std |
|---|---|---|
| RED | 150.73 | 39.72 |
| GREEN | 123.76 | 34.06 |
| BLUE | 92.58 | 30.09 |
| DSM_NORMALIZED | −9.73 | 4.68 |

Applied in `src/dataset.py` before feeding to the model: `normalized = (raw - mean) / std`.
