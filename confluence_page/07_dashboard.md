# 7. The Evaluation Dashboard

A Streamlit application provides interactive, per-tile evaluation across all registered models. It reads pre-computed metrics parquet files and supports on-demand tile visualization.

`[IMAGE PLACEHOLDER: Full dashboard screenshot showing the tab bar (Statistics / Overview / Tile explorer / Map / Compare models / Test set / Raw data) and a model selector. Run: streamlit run DASHBOARD_FROM_TYTONAI/app.py from repo root.]`

---

## 7.1 Architecture

| Component | Implementation |
|---|---|
| UI framework | Streamlit |
| Query engine | DuckDB (in-process, reads parquet) |
| Model registry | `models_registry.json` + `models/*.pth` |
| Train metrics | `output/metrics_<model>.parquet` |
| Test metrics | `output/test_metrics_<model>.parquet` |
| Tile imagery | `.npz` files (NPZ → PNG generated on-demand, cached) |
| Geo data | `output/tiles_geo_<dataset>.parquet` (lat/lon, site name, pixel size) |

Selecting a model in the sidebar loads its metrics into DuckDB. All tabs query the same in-memory tables. Switching models is instant — no re-loading.

---

## 7.2 Tab Descriptions

### Statistics

Statistical analysis of the erosion distribution and spatial structure. Does not depend on model selection.

- **Lorenz curve** — cumulative % of erosion pixels vs % of tiles (sorted poorest → richest). Shows concentration with Gini index.
- **Cumulative precision / recall / F1** — how metrics evolve as you include progressively denser tiles
- **Spatial variogram** — empirical variogram + fitted spherical model + DBSCAN site map

`[IMAGE PLACEHOLDER: Statistics tab — Lorenz 2×2 subplot (Lorenz curve, cumulative precision, recall, F1). Source: dashboard Statistics tab with any model loaded.]`

---

### Overview

Global KPIs + distribution analysis for the selected model on the training set.

- **KPI strip:** tiles evaluated, tiles with erosion, mean F1 (macro), global F1 (micro), recall, F1 no-erosion
- **Filtered KPI strip:** same metrics for tiles above a minimum erosion pixel threshold (slider)
- **Precision vs Recall scatter:** one point per tile, colored by F1, sized by erosion pixel count
- **Metric distributions:** histograms of F1 / Precision / Recall / F1 no-erosion (train, can overlay test)
- **Worst/Best-10 lists:** worst/best 10 tiles by F1, by precision, by recall, by F1 no-erosion

---

### Tile Explorer

Fully filterable, sortable tile table with click-to-visualize.

- Sort by: worst/best F1, precision, recall, most/least erosion pixels (17 sort options)
- Filters: F1 range, min erosion pixels, min tile size, max erosion pixels, predicted erosion pixels, tile name search
- Click any row → generates side-by-side visualization:
  - **Side-by-side masks:** ground truth mask vs model prediction
  - **Contour overlay:** prediction contours overlaid on RGB and DSM channels
- Displays per-tile metrics: F1, Precision, Recall, IOU, pixel counts

`[IMAGE PLACEHOLDER: Tile Explorer — example tile visualization: left panel = RGB+GT mask, right panel = RGB+predicted mask. Shows a correctly detected erosion patch. Choose a high-F1 erosion tile from the training set.]`

---

### Map

Geospatial scatter of all training tiles, colored by metric.

- Color-by options: F1 erosion, Precision erosion, Recall erosion, erosion pixel count, pixel size (m)
- Filters: min erosion pixels, F1 range
- Test set overlay (purple, togglable)
- Jump-to-tile: paste tile UUID to highlight and center
- Click tile on map → same visualization as Tile Explorer

`[IMAGE PLACEHOLDER: Map tab — training tiles colored by F1 erosion (RdYlGn colorscale), test zones visible as purple cluster in 3 geographic locations (Dampier, Marshall, Dugong).]`

---

### Compare Models

Cross-model leaderboard and deep analysis.

**Leaderboard:** all registered models, ranked by selected metric (default: test F1). Shows train + test columns side-by-side. MLflow models flagged with 🔬. v2 shown with ⚠️ annotation.

**F1 Distributions:** box plots and histograms, one curve per model. Supports dataset filter (train / test / both overlay).

**Pairwise Deep-Dive:** select 2 models:
- Scatter: Model A F1 vs Model B F1 per tile (who wins, by how much, sized by erosion px)
- Histogram: Δ F1 distribution
- Counts: A wins / B wins / Ties (at chosen threshold)

**Erosion Density Gap:** tiles bucketed by erosion coverage (None / Sparse 0–5% / Medium 5–25% / Dense >25%):
- Count and % of test set per bucket
- Mean F1 per bucket per model (bar chart)
- Contribution to total F1 gap (which density tier drives the difference between two models)

`[IMAGE PLACEHOLDER: Compare Models leaderboard — test F1 bar chart, all models. MiT-B3 and v3_ep399 at top, v2 with leakage warning, UNet at bottom.]`

---

### Test Set Explorer

Mirror of Tile Explorer, restricted to the 1,766 test tiles (purple theme).

Same filters and visualization as Tile Explorer. Adds a "filter by capture" option (Dampier / Marshall / Dugong).

---

### Raw Data

- Full parquet viewer (first 5,000 rows sorted by F1 ascending)
- DuckDB SQL console:
  ```sql
  -- Example: find test tiles where MiT-B3 has low recall
  SELECT imagery_file, f1_erosion, recall_erosion, n_erosion_pixels
  FROM test_metrics
  WHERE recall_erosion < 0.3 AND n_erosion_pixels > 1000
  ORDER BY recall_erosion ASC
  LIMIT 20
  ```

---

## 7.3 How to Run

```bash
# From repo root
streamlit run DASHBOARD_FROM_TYTONAI/app.py
```

Requires:
- Model weights in `models/` (symlinked from checkpoints or downloaded)
- Metrics parquet in `output/` (generated by `python -m src.evaluate --model-path models/<model>.pth`)
- Test metrics in `output/test_metrics_<model>.parquet` (generated by `python -m DASHBOARD_FROM_TYTONAI.evaluate_test --model <model>.pth`)
