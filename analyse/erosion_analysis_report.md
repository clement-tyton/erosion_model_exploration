# Erosion Model Analysis Report

*Model: v3_split_test — epoch 1819 | 19,752 tiles total | 6,852 tiles with erosion (34.7%)*

---

## 1. Dataset Overview

| Statistic | Value |
|---|---|
| Total tiles evaluated | 19,752 |
| Tiles with erosion (≥1 px) | 6,852 (34.7%) |
| Total erosion pixels | 98,468,414 |
| Median tile footprint | 8.1 m × 8.1 m |
| Median GSD | ~2.1 cm/px (UAV at ~70 m altitude) |

---

## 2. Global Performance

| Metric | Value |
|---|---|
| Overall Precision (erosion) | **94.7%** |
| Overall Recall (erosion) | **86.9%** |
| Overall F1 (erosion) | **90.6%** |
| Gini index (erosion distribution) | **0.53** |

**The binding constraint is recall, not precision.** Precision is near its ceiling at 94.7%.
Any intervention that improves recall improves F1 proportionally.

---

## 3. Lorenz Curve — Erosion Concentration

Tiles are sorted by ascending erosion pixel count (poorest → richest).

| Tile percentile (poorest → richest) | Cumulative % of erosion pixels |
|---|---|
| Bottom 10% | 0.3% |
| Bottom 20% | 1.4% |
| Bottom 30% | 3.6% |
| Bottom 50% | 13.0% |
| Bottom 70% | 31.6% |
| Bottom 90% | 65.2% |

**Gini index = 0.53** — significant concentration. The bottom 50% of erosion tiles hold only
13% of all erosion pixels. These are "fringe tiles": they sit at the spatial boundary of an
erosion patch and capture only a handful of pixels from it. True erosion cores are
concentrated in the top ~30% of tiles by density.

**What this means for metrics:** aggregate F1 is dominated by the dense tiles. The fringe
tiles barely move the needle on F1 or precision at aggregate level, but they are where recall
structurally collapses — and they represent the majority of erosion-containing tiles.

---

## 4. Cumulative Precision — Fast Rise, then Plateau

**Shape:** Spikes from near 0% to ~80% in the first 5% of tiles, then climbs smoothly to 94.7%.

**Why the initial spike looks alarming but is not:**
The first tiles contain 1–10 actual erosion pixels. A single false-positive pixel (1 wrongly
classified pixel out of ~147,000) collapses the per-tile precision. Summed across the first N
tiles, the cumulative denominator (TP + FP) is tiny and a few false positives dominate. These
tiles hold < 0.3% of all erosion — this zone is statistically meaningless.

**What the plateau says:**
Once past that threshold, precision is already 85–90% and climbs smoothly to 94.7%.
For any tile where erosion is meaningfully present, what the model calls erosion is almost
always truly erosion. **The model is highly discriminative; it very rarely invents erosion.**

---

## 5. Cumulative Recall — Strictly Increasing, Near-Linear

**Shape:** Recall is strictly increasing across the full sorted range of tiles, near-linearly.

This outcome was **not guaranteed.** If the model were overwhelmed by large erosion patches
(many false negatives precisely where erosion is dense), recall would have *decreased* as
dense tiles were added. The monotone increase is a non-trivial finding.

**Recall at cut points:**
| Include up to (% of tiles) | Cumulative global recall |
|---|---|
| 30% of tiles | 40.0% |
| 50% of tiles | 61.6% |
| 70% of tiles | 73.7% |
| 100% of tiles | 86.9% |

**Mechanism 1 — Detection threshold effect:** A tile with 5 scattered erosion pixels at a
field boundary generates a weak, diffuse activation signal that never crosses the
classification threshold. More erosion → stronger, more coherent activation → more pixels recalled.

**Mechanism 2 — Texture vs. structure:** Erosion is recognisable as a *spatial pattern*
(bare soil texture, slope context, connected patches), not a pixel-by-pixel feature.
When enough erosion is present to form that structure within the receptive field, the model
sees it confidently. When it is just a 5-pixel fringe, the context is ambiguous.

**The near-linear shape** implies the marginal recall gain per added tile is roughly constant
across the full range — no cliff, no saturation. The model has not yet found a hard upper
bound on recall; there is room to improve.

---

## 6. Spatial Autocorrelation — Variogram

### Setup

Tiles were merged with their geographic coordinates (`x_center`, `y_center`).
DBSCAN (ε = 200 m, min_samples = 2) grouped them into **48 geographic sites** (0 isolated tiles).
The empirical variogram was computed using **within-site pairs only**, avoiding cross-site
comparisons that would mix distinct erosion fields.

For each distance lag h, the semi-variance is:

```
γ(h) = 0.5 × mean( (n_erosion_i − n_erosion_j)² )   over all pairs at distance h ± Δh/2
```

Each lag bin held **40,000–120,000 pairs** → highly stable estimates.

### Results

| Parameter | Value | Interpretation |
|---|---|---|
| Nugget | 5.07 × 10⁷ px² | √nugget = 7,120 px — irreducible local noise |
| Sill | 1.17 × 10⁸ px² | √sill = 10,817 px — spatially structured variation |
| **Range** | **30.9 m** | Erosion patch characteristic diameter |
| Total semi-deviation | √(nugget+sill) = **12,894 px** | Total variability at independence |

**Structured vs noise variance:**
- 69.8% of total variance is spatially structured (within the range)
- 30.2% is pure local noise (nugget)

The variogram shows that erosion is far from random: patches have a real spatial signature
with a characteristic diameter of ~31 m.

**Reading √γ(h) on the y-axis:**
√γ(h) is the *standard deviation of the difference* in erosion pixel count between two tiles
at distance h — expressed in pixels, directly interpretable.
- At h → 0: two tiles still differ by ~7,120 px on average (nugget noise)
- The difference grows as tiles become more dissimilar
- At h ≥ 31 m: tiles are fully independent, the curve flattens at √γ = 12,894 px

---

## 7. Tile Size Mismatch — Root Cause of Recall Deficit

```
Autocorrelation range  :  30.9 m
Current tile footprint :   8.1 m
Ratio                  :  30.9 / 8.1 ≈ 3.8×
```

The current tile sees only **26% of a typical erosion patch diameter.** Most tiles that
contain any erosion are fringe tiles — they lie at the edge of a 30.9 m patch and capture
only a sliver of it. That is precisely the population where recall collapses (40% at the
30th percentile).

### Recommended tile size

To give the model a field of view that covers at least one full erosion patch:

```
Required pixels = 30.9 m ÷ 0.021 m/px = 1,471 px

Round to next multiple of 384:
  → 1,536 px  (= 4 × 384)

Verification:
  1,536 px × 0.021 m/px = 32.3 m ≥ 30.9 m  ✓
```

| GSD (cm/px) | Pixels needed for 30.9 m | Nearest candidate |
|---|---|---|
| 1.8 (finest) | 1,717 px | 1,792 px (≈ 4.67 × 384) |
| 2.1 (median) | 1,471 px | **1,536 px (= 4 × 384)** |
| 2.5 (coarse) | 1,236 px | 1,280 px (= 3.3 × 384) |

**Recommended tile size: 1,536 × 1,536 px** — robust across the full GSD distribution.

---

## 8. Conclusions

| Finding | Evidence | Implication |
|---|---|---|
| Recall (86.9%) is the sole binding constraint | F1 = 90.6% vs Precision = 94.7% | Focus all effort on recall improvement |
| Recall grows near-linearly with erosion density | Cumulative recall curve | No architectural ceiling yet — room to improve |
| Erosion is spatially concentrated (Gini 0.53) | Bottom 50% tiles → only 13% of erosion | Fringe tiles dominate recall deficit |
| Erosion patch diameter = 30.9 m | Variogram range | Characteristic spatial scale |
| Current tile (8.1 m) = 26% of patch diameter | Tile footprint vs range | Most tiles show only fragments |
| 69.8% of variance is spatially structured | Sill / (nugget + sill) | Tile size increase will directly help |

**The model is not fundamentally wrong — it is operating on tiles that are too small.**
It reliably identifies erosion when a coherent patch is visible in the receptive field.
When the tile shows only a thin fringe, the spatial cues are absent and recall collapses.

**Highest-leverage intervention before any architectural change:**
increase tile size to **1,536 × 1,536 px**, directly addressing the root cause identified
by the variogram analysis.
