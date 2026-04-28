# Erosion Detection — Executive Summary

## What This Project Is About

This project covers the full R&D cycle for an automatic erosion detection model applied to UAV imagery at TytonAI. Starting from an existing baseline model, the work spans training methodology, rigorous evaluation design, architecture benchmarking, and the development of a dedicated interactive dashboard.

The core task is binary semantic segmentation: classify every pixel of a 4-band raster (Red, Green, Blue, DSM_Normalized) as either *erosion* or *no erosion*.

---

## What Was Found on Arrival

The initial model (v1, Jaswinder's baseline) had a significant false-positive problem: it classified many patches as erosion that were not. An early iteration (v2) corrected this by adding no-erosion training data, bringing false positives down. However, neither model was evaluated on a geographically isolated test set — the performance numbers reported at the time were not a reliable measure of generalization.

---

## What Was Built

**A rigorous geographic train/test split** was established. Three capture zones — Dampier, Marshall, and Dugong — were held out entirely from training. No tile from these zones was used in any model from v3 onward.

**Multiple models were trained**, first via TytonAI's ObjectTrain pipeline, then replicated via MLflow with architectural variants (UNet and SegFormer).

**A statistical evaluation dashboard** was built on top of per-tile metrics, allowing fine-grained inspection of every model across all tiles: Lorenz curves, spatial autocorrelation analysis, per-tile visualization (ground truth vs prediction), geographic maps, and model comparison leaderboards.

---

## Results at a Glance

| Model | Test F1 | vs. Baseline | Note |
|---|---|---|---|
| v3 ep50 — TytonAI baseline | 0.517 | — | First properly split model |
| v3 ep399 — longer training | 0.606 | +17% | With early stopping |
| **SegFormer MiT-B3 ep200** | **0.630** | **+22%** | Best architecture, MLflow |

*Test set: 1,766 tiles from Dampier, Marshall, and Dugong — never seen during training.*

---

## Key Insight: Why Global F1 Is Not Enough

Erosion is distributed very unequally across tiles (Gini index = 0.53). The bottom half of erosion tiles contain only 13% of all erosion pixels. A model that captures one big erosion patch can look great on aggregate F1 without actually generalizing well. The dashboard addresses this with Lorenz curves, density-bucket decomposition, and per-tile statistics.

---

## The Structural Bottleneck

Spatial autocorrelation analysis (variogram) shows that erosion patches have a characteristic diameter of ~31 m. The current tile footprint is 8.1 m — only 26% of a patch. Most "erosion tiles" are fringe tiles capturing only a sliver of a patch, which is exactly where recall collapses. The highest-leverage next step is increasing tile size to **1,536 × 1,536 px** (≈ 32 m), before any further architecture changes.

---

## What Comes Next

- AMP + torch.compile training variants (near-factor-2 speedup, currently running)
- Tile size increase to 1,536 px and retrain MiT-B3
- Consolidation of training into TytonAI exclusively (MLflow replication was a transitional approach)
