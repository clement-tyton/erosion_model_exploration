# 3. First-Level Analysis: Erosion Statistics

> **Why this section matters:** global F1 alone is a misleading metric for this dataset. This section explains why, with data.

---

## 3.1 Erosion is Highly Concentrated (Lorenz Curve)

Tiles sorted by ascending erosion pixel count (poorest → richest in erosion content):

| Tile percentile (poorest → richest) | Cumulative % of all erosion pixels |
|---|---|
| Bottom 10% | 0.3% |
| Bottom 20% | 1.4% |
| Bottom 30% | 3.6% |
| Bottom 50% | **13.0%** |
| Bottom 70% | 31.6% |
| Bottom 90% | 65.2% |

**Gini index = 0.53** — significant spatial concentration. The bottom half of erosion-containing tiles holds only 13% of all erosion pixels. These are "fringe tiles": they sit at the edge of a large erosion patch and capture only a handful of pixels from it.

### What This Means for F1

Imagine a model that perfectly detects one large erosion patch that happens to cover 90% of all erosion pixels across the dataset. Its global F1 will look very high — yet it completely misses the majority of erosion-containing tiles. Conversely, a model that misses that single large patch will look very poor even if it correctly classifies thousands of smaller ones.

**Global F1 is dominated by the dense tiles.** It does not measure breadth of detection. This is why the dashboard complements F1 with:
- Cumulative precision/recall curves (sorted by erosion density)
- Density-bucket decomposition (none / sparse / medium / dense)
- Per-tile metric distributions

`[IMAGE PLACEHOLDER: lorenz_curves.png — 2×2 subplot: Lorenz curve (top left), cumulative precision (top right), cumulative recall (bottom left), cumulative F1 (bottom right). File exists at analyse/lorenz_curves.png]`

### Reading the Cumulative Curves

**Precision:** spikes from near 0 to ~80% in the first 5% of tiles (fringe tiles with 1–10 erosion pixels — statistically meaningless, dominated by 1–2 false-positive pixels), then climbs smoothly to 94.7%. The initial spike looks alarming but represents < 0.3% of all erosion. Where erosion is meaningfully present, what the model predicts as erosion is almost always genuinely erosion.

**Recall:** strictly increasing and near-linear across the full range. No cliff. No saturation. The model does not become "overwhelmed" by dense tiles — recall grows proportionally. This means there is no architectural ceiling yet; recall can still be improved.

---

## 3.2 Spatial Autocorrelation — Variogram Analysis

### Setup

Each tile has geographic coordinates (`x_center`, `y_center` in metres). DBSCAN clustering (ε = 200 m, min_samples = 2) grouped tiles into **48 geographic sites** with 0 isolated tiles.

An empirical variogram was computed using **within-site pairs only** — cross-site pairs would conflate distance effects with between-site erosion differences.

For each distance lag h:
```
γ(h) = 0.5 × mean( (n_erosion_i − n_erosion_j)² )   for all pairs at distance h ± Δh/2
```

Each lag bin contained 40,000–120,000 pairs → highly stable estimates.

### Results

| Parameter | Value | Interpretation |
|---|---|---|
| Nugget | 5.07 × 10⁷ px² | √nugget = 7,120 px — irreducible local noise |
| Sill | 1.17 × 10⁸ px² | √sill = 10,817 px — spatially structured variance |
| **Range** | **30.9 m** | Erosion patch characteristic diameter |
| Structured variance | **69.8%** | Most variance is spatially organized |

The range of 30.9 m is the distance at which two tiles become statistically independent — it represents the characteristic diameter of an erosion patch.

`[IMAGE PLACEHOLDER: variogram.png — top: empirical variogram scatter + fitted spherical model (nugget, sill, range labeled); bottom: DBSCAN site map with 48 clusters colored by site ID. File exists at analyse/variogram.png]`

### Tile Size Mismatch — Root Cause of Recall Deficit

```
Erosion patch diameter  :  30.9 m
Current tile footprint  :   8.1 m
Ratio                   :  30.9 / 8.1 ≈ 3.8×
Current tile = 26% of a typical patch
```

The current tile sees less than one quarter of a typical erosion patch. The majority of erosion-positive tiles are fringe tiles — they lie at the edge of a 30.9 m patch and capture only a sliver. This is precisely the population where recall collapses.

### Recommended Tile Size

```
Required pixels = 30.9 m ÷ 0.021 m/px = 1,471 px
Round to next power-of-2-friendly multiple of 384:
→ 1,536 px  (= 4 × 384)
→ footprint = 1,536 × 0.021 = 32.3 m ≥ 30.9 m  ✓
```

| GSD (cm/px) | Pixels needed | Nearest candidate |
|---|---|---|
| 1.8 (finest) | 1,717 px | 1,792 px |
| 2.1 (median) | 1,471 px | **1,536 px** |
| 2.5 (coarse) | 1,236 px | 1,280 px |

**Recommended tile size: 1,536 × 1,536 px** — robust across the full GSD range.

This is the highest-leverage intervention before any architecture change. The model is not fundamentally wrong; it is operating on tiles that are too small to show it a coherent erosion patch.
