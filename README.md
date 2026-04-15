# Erosion R&D — Evaluation & Exploration Dashboard

Internal tool for evaluating and comparing erosion detection models trained on drone imagery (Western Australia).  
Built over several sessions by Clément + Claude.

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
├── models/                          # Model checkpoints (.pth)
│   ├── model_v1_jaswinder_epoch50.pth
│   ├── model_v2_no_erosion_td_epoch50.pth
│   ├── model_v3_split_test_epoch50.pth
│   ├── model_v3_split_test_epoch78.pth
│   └── model_v3_split_test_epoch80.pth
│
├── models_registry.json             # Source of truth: model → dataset mapping
│
├── data/                            # Downloaded tile NPZ files (one dir per dataset)
│   ├── v1_jaswinder/
│   ├── v2_no_erosion_td/
│   ├── v3_split_test_ep50/
│   └── v3_split_test_final/         # shared by ep78 + ep80
│
├── <dataset_name>.json              # Tile manifest for each dataset (downloaded from S3)
│   ├── v1_jaswinder.json
│   ├── v2_no_erosion_td.json
│   ├── v3_split_test_ep50.json
│   └── v3_split_test_final.json
│
├── output/                          # Generated outputs
│   ├── metrics_<model_stem>.parquet # Per-tile metrics (one file per model)
│   ├── tiles_geo.parquet            # Geographic centroids for all tiles (WGS84)
│   └── tiles/                       # Generated PNG visualisations (one dir per model)
│
└── src/
    ├── config.py                    # Paths, model hyperparameters, normalisation stats
    ├── dataset.py                   # NPZ loading, band selection, normalisation, DataLoader
    ├── model.py                     # SMP UNet builder + checkpoint loading
    ├── evaluate.py                  # Batch inference → per-tile metrics → Parquet
    ├── visualize.py                 # PNG generation (side-by-side masks + contour overlay)
    ├── build_geo.py                 # Extract WGS84 centroids from all NPZ tiles
    ├── download_tiles.py            # Download NPZ files from S3 given a manifest JSON
    ├── run_all.py                   # Full pipeline: download → evaluate for all models
    └── app.py                       # Streamlit dashboard
```

---

## Model registry

`models_registry.json` is the single source of truth linking each model to its training dataset.

```json
{
  "model_file":    "model_v3_split_test_epoch80.pth",
  "version":       "v3",
  "epoch":         80,
  "description":   "v2 + no-erosion TD + split test — epoch 80",
  "dataset_name":  "v3_split_test_final",       ← human-readable dataset key
  "tiles_json_id": "ef1410ef-...",               ← S3 UUID of the manifest
  "s3_model":      "s3://..."
}
```

- `dataset_name` drives both the local manifest filename (`v3_split_test_final.json`) and the tile directory (`data/v3_split_test_final/`)
- Models sharing the same `dataset_name` reuse the same downloaded tiles (no duplicate download)

---

## Model versions

| Model | Dataset | Description |
|---|---|---|
| v1 Jaswinder ep50 | v1_jaswinder | Original baseline (Jaswinder) |
| v2 ep50 | v2_no_erosion_td | v1 + added no-erosion training data |
| v3 ep50 | v3_split_test_ep50 | v2 + train/test split, early checkpoint |
| v3 ep78 | v3_split_test_final | Same as ep80, earlier checkpoint |
| v3 ep80 | v3_split_test_final | Best checkpoint from final training run |

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
This will for each model, in order:
1. Download the tile manifest JSON from S3
2. Download all NPZ tiles into `data/<dataset_name>/` (32 parallel workers)
3. Run inference and save `output/metrics_<model>.parquet`

Already-computed parquets are skipped. Use `--force` to re-evaluate.

```bash
python -m src.run_all --dry-run    # preview what would run
python -m src.run_all --force      # force re-evaluation (tiles not re-downloaded)
```

### 4. Build geographic index (once)
```bash
python -m src.build_geo
```
Extracts WGS84 centroids from all NPZ tiles → `output/tiles_geo.parquet`.  
Required for the Map tab.

### 5. Launch the dashboard
```bash
streamlit run src/app.py
```

---

## Dashboard tabs

### Overview
Global KPIs for the selected model:
- **Mean F1 erosion** — macro: average of per-tile F1 on erosion tiles
- **Global F1 erosion** — micro: 2·ΣTP / (2·ΣTP + ΣFP + ΣFN) across all pixels
- Precision / Recall / F1 no-erosion
- A slider filters the KPIs and charts to tiles with ≥ N erosion pixels (useful to exclude very sparse tiles where detection is inherently hard)
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
- Summary table: F1, global F1, precision, recall, IOU, dataset, tile count, erosion prevalence
- Bar chart comparing any metric across models
- Scatter: erosion prevalence in dataset vs model F1 (helps understand data impact on performance)

### Raw data
- Full parquet viewer (first 5 000 rows)
- DuckDB SQL console — query the `metrics` view directly

---

## Key design decisions

**Why Parquet + DuckDB?**  
19 752 tiles per model × 5 models. Parquet gives fast columnar reads; DuckDB queries it in-process with SQL — no server, no loading the whole file into RAM.

**Why inference on CPU in the app?**  
CUDA causes segfaults in Streamlit's forked worker process. CPU is slower but stable for on-the-fly single-tile inference. Batch evaluation (`run_all`) uses CUDA.

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

Each tile is a `.npz` archive with keys:
- `RED`, `GREEN`, `BLUE`, `DSM`, `DSM_NORMALIZED`, `MEP` — band arrays (H × W float32)
- `SRID` — EPSG code (20350 = GDA94 / MGA zone 50, Western Australia)
- `GEO_TRANSFORM` — 9-element row-major 3×3 affine matrix:  
  `[scale_x, shear_x, x_origin, shear_y, scale_y, y_origin, 0, 0, 1]`
- `VERSION` — format version

Mask tiles use the same format with a single band containing pixel labels:
- `0` → background (ignored in metrics)
- `1` → no-erosion
- `14` → erosion

---

## Normalisation

The model was trained with per-band mean/std normalisation (values in 0–255 scale):

| Band | Mean | Std |
|---|---|---|
| RED | 150.73 | 39.72 |
| GREEN | 123.76 | 34.06 |
| BLUE | 92.58 | 30.09 |
| DSM_NORMALIZED | −9.73 | 4.68 |

Applied in `src/dataset.py` before feeding to the model.
