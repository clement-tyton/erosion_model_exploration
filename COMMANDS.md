# Commands — Erosion R&D dashboard

All commands run from the **repository root** (`erosion_R_and_D/`).

---

## 1 · Evaluate models on the **train set**

```bash
# Full pipeline for all models: download tiles + run inference + write metrics
python -m src.run_all

# Force re-evaluation (tiles not re-downloaded, metrics overwritten)
python -m src.run_all --force

# Dry run — show status without any I/O
python -m src.run_all --dry-run

# Control download parallelism (default 32)
python -m src.run_all --workers 16
```

Outputs: `output/metrics_<model_stem>.parquet` (one file per model)

---

## 2 · Evaluate models on the **test set**

```bash
# Full pipeline: download test tiles + build geo index + evaluate all models
python -m DASHBOARD_FROM_TYTONAI.evaluate_test

# Force re-evaluation (overwrite existing parquets)
python -m DASHBOARD_FROM_TYTONAI.evaluate_test --force

# Evaluate a single model only (example)
python -m DASHBOARD_FROM_TYTONAI.evaluate_test --model model_v3_split_test_epoch_95.pth

# Skip steps you've already done
python -m DASHBOARD_FROM_TYTONAI.evaluate_test --skip-download   # tiles already present
python -m DASHBOARD_FROM_TYTONAI.evaluate_test --skip-geo        # geo parquet already built
python -m DASHBOARD_FROM_TYTONAI.evaluate_test --skip-eval       # just rebuild geo

# Control download parallelism (default 32)
python -m DASHBOARD_FROM_TYTONAI.evaluate_test --workers 16
```

Outputs:
- `output/tiles_geo_test.parquet` — WGS84 centroids for all 1 766 test tiles (built once)
- `output/test_metrics_<model_stem>.parquet` — per-tile metrics on the test set

---

## 3 · Launch the **dashboard**

```bash
streamlit run DASHBOARD_FROM_TYTONAI/app.py
```

Then open `http://localhost:8501` in your browser.

---

## Typical first-time workflow

```bash
# Step 1 — evaluate all models on train set (skip if parquets already exist)
python -m src.run_all

# Step 2 — evaluate all models on test set
python -m DASHBOARD_FROM_TYTONAI.evaluate_test

# Step 3 — launch dashboard
streamlit run DASHBOARD_FROM_TYTONAI/app.py
```
