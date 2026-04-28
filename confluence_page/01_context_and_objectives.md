# 1. Context & Objectives

## Task Definition

Erosion detection is formulated as a **2-class semantic segmentation** problem:

- **Class 1:** no erosion
- **Class 14:** erosion

Each tile is a 4-band raster:

| Band | Description |
|---|---|
| RED | Red reflectance |
| GREEN | Green reflectance |
| BLUE | Blue reflectance |
| DSM_NORMALIZED | Normalized Digital Surface Model (elevation) |

Imagery is acquired by UAV at approximately 70 m altitude, yielding a ground sampling distance (GSD) of ~2.1 cm/pixel.

---

## Objectives

1. **Establish a reproducible evaluation methodology** with a geographically isolated test set (no data leakage between train and test).
2. **Train and benchmark multiple architectures** (UNet, SegFormer) and track results in MLflow.
3. **Characterize the dataset statistically** — erosion distribution, spatial autocorrelation, tile size adequacy.
4. **Build an interactive dashboard** enabling fine-grained per-tile exploration and cross-model comparison.
5. **Identify the highest-leverage improvements** for future iterations.

---

## Scope

This work operates outside the TytonAI platform for the training and evaluation components. Data is extracted from TytonAI as NPZ tiles via the ObjectTrain/TestEpoch workflow outputs; models are trained locally and tracked via MLflow; evaluation metrics are computed tile-by-tile and stored as parquet files consumed by the Streamlit dashboard.

> **This is the last project of this hybrid kind.** The medium-term direction is to consolidate all experimentation within TytonAI only. The methodology built here (balanced tiles JSON, per-tile metrics, geographic split) feeds directly into that longer-term goal.

---

`[IMAGE PLACEHOLDER: Excalidraw schema — TytonAI platform → ObjectTrain → balanced_tiles.json (NPZ extraction) → Local training (MLflow) → Evaluation dashboard]`
