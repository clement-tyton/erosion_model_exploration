# Lorenz Analysis — Methodology, Interpretation & Conclusions

*Model: v3_split_test — epoch 1819 | 19,752 tiles total | 6,852 tiles with erosion*

---

## 1. What Are We Measuring?

The model produces a **binary segmentation mask** for each tile: every pixel is either
classified as erosion (class 14) or not. For each tile we record four pixel-level counts:

| Symbol | Meaning |
|---|---|
| TP | True positives — erosion pixels correctly predicted as erosion |
| FP | False positives — non-erosion pixels wrongly predicted as erosion |
| FN | False negatives — erosion pixels missed by the model |
| n\_erosion\_pixels | Total ground-truth erosion pixels in the tile (= TP + FN) |

---

## 2. How the Curves Are Built

### Step 1 — Filter and sort

All tiles with **zero erosion pixels are excluded** (no ground truth to evaluate against).
The remaining 6,852 tiles are then **sorted by ascending `n_erosion_pixels`** — from
the tile with the single loneliest erosion pixel up to the tile with the most.

This ordering is the foundation of all four curves: x-axis position = "we have included
the first N% of tiles by erosion density."

### Step 2 — Lorenz curve

```
lorenz_y[N] = Σ n_erosion_pixels[:N]  /  Σ n_erosion_pixels_all  × 100
```

At each point N, what fraction of the **total erosion area** (pixel count) is held by
the N% least erosion-dense tiles?

### Step 3 — Cumulative global precision / recall / F1

At each cut-point N, precision and recall are computed as **micro (pixel-level) averages**
over the first N tiles — i.e. the TP/FP/FN counts are summed across all N tiles before
dividing:

```
Precision[N] = Σ TP[:N]  /  ( Σ TP[:N] + Σ FP[:N] )
Recall[N]    = Σ TP[:N]  /  ( Σ TP[:N] + Σ FN[:N] )
F1[N]        = 2 × Precision[N] × Recall[N]  /  ( Precision[N] + Recall[N] )
```

This is **not** the mean of per-tile metrics. It is the single global metric as if all
the first N tiles were one big image. The last point (N = 100% of tiles) is therefore
identical to the overall model metric reported in any standard evaluation.

---

## 3. Interpreting Each Curve

### 3.1 Lorenz Curve — Erosion Inequality

The Lorenz curve borrowed from economics measures how **unequally erosion is distributed**
across tiles. A diagonal line (x = y) would mean every tile contains the same fraction
of the total erosion — perfect equality. The further the curve bends below the diagonal,
the more concentrated erosion is.

**Reading our curve:**

- The bottom 50% of tiles (by erosion count) hold only ~13% of all erosion pixels.
- The top 10% of tiles hold the large majority of erosion area.
- **Gini index ≈ 0.7** — extreme inequality, comparable to highly unequal income
  distributions.

**What this means practically:**  
Most erosion-containing tiles are "trace tiles" — they sit at the spatial fringe of an
erosion patch and capture only a handful of pixels from it. True erosion cores are
concentrated in a small fraction of the dataset. Any aggregate metric is therefore
dominated by these dense tiles; the fringe tiles barely move the needle on F1 or
precision, but they are where recall structurally collapses.

### 3.2 Precision Curve — Fast Rise then Plateau

**Shape:** Spikes from near zero to ~80–85% in the first 5% of tiles, then climbs
smoothly to 94.7%.

**Why the initial spike looks alarming:**  
The first tiles contain 1–10 actual erosion pixels. A single false-positive pixel
(1 wrongly classified pixel out of 147,000) collapses the per-tile precision to near
zero. But summed across the first N tiles, the cumulative denominator (TP + FP) is
tiny — a few false positives dominate, creating an artefactual drop. The Lorenz curve
tells us these tiles hold < 0.3% of all erosion, so this zone is statistically
meaningless.

**What the plateau actually says:**  
Once past that threshold, precision is already 85–90% and rises smoothly to 94.7%.
For any tile where erosion is meaningfully present, what the model calls erosion is
almost always erosion. **The model is highly discriminative; it very rarely invents
erosion out of nothing.**

### 3.3 Recall Curve — Strictly Increasing, Near-Linear

**Shape:** Recall is strictly increasing across the full sorted range of tiles, nearly
linearly.

This outcome was **not guaranteed**. If the model were overwhelmed by large erosion
patches (many false negatives precisely where erosion is dense), recall would have
*decreased* as we added erosion-rich tiles. The monotone increase is a non-trivial
finding.

**Mechanism 1 — Detection threshold effect:**  
A tile with 5 scattered erosion pixels at a field boundary generates a weak, diffuse
activation signal. The model's confidence never crosses the classification threshold.
More erosion pixels → stronger, more spatially coherent activation → more pixels recalled.

**Mechanism 2 — Texture vs. structure:**  
Erosion is recognisable as a *spatial pattern*: bare soil texture, slope context,
connected patches. When enough erosion is present to form that structure within the
receptive field, the model sees it confidently. When it is just a 5-pixel fringe,
the local context is ambiguous.

**The near-linear shape** implies that the marginal recall gain per added tile is
roughly constant across the sorted range — no cliff, no early saturation. The model
has not yet found a hard upper bound on recall; there is room to improve.

### 3.4 F1 Score — Recall Is the Binding Constraint

**Shape:** Mirrors precision at the low end, then converges to recall's slope.

The overall F1 = 90.6% is pulled down entirely by recall (86.9%). Precision at 94.7%
is nearly at its ceiling. **Any intervention that improves recall improves F1
proportionally; further gains on precision have diminishing returns.**

---

## 4. Root Cause Hypothesis — Tile Size Mismatch

The recall curve's near-linear growth with erosion density, combined with the extreme
Lorenz inequality, points to a single structural explanation: **the current tile footprint
is much smaller than the characteristic size of an erosion patch.**

A tile that straddles the boundary of an erosion field gives the model a confusing
context — half bare soil, half vegetated, no coherent erosion structure. These boundary
tiles dominate the lower 30–50% of the sorted distribution. A model trained on
384 × 384 px tiles learns to recognise erosion from fragments of it, not from full patches.

This hypothesis is tested quantitatively via the empirical variogram.

---

## 5. Spatial Autocorrelation — Variogram Methodology

### What the variogram measures

For every pair of tiles (i, j) within the same geographic site, separated by distance h:

```
γ(h) = 0.5 × mean( (n_erosion_i − n_erosion_j)² )   for all pairs at distance h ± Δh/2
```

- **Small h**: nearby tiles tend to have similar erosion counts → (z_i − z_j)² ≈ 0 → γ low
- **Growing h**: tiles become more dissimilar → γ rises
- **Beyond the range**: tiles are statistically independent → γ plateaus at the **sill**

### Geographic clustering with DBSCAN

Tiles are first grouped into geographic sites using DBSCAN (ε = 200 m, min\_samples = 2).
This produces 48 sites. Only **within-site pairs** are used to compute the variogram —
this eliminates cross-site comparisons that would mix distinct erosion fields.

### Weighted average across sites

For each lag bin h, the final γ(h) is a **pair-count-weighted average across sites**:

```
γ_total(h) = Σ_sites [ n_pairs_site(h) × γ_site(h) ]
           / Σ_sites [ n_pairs_site(h) ]
```

Sites with many tiles contribute more to the estimate. With 30 valid lag bins each
containing 40,000–120,000 pairs, the estimate is highly stable.

### Spherical model fit

A standard spherical model is fitted by nonlinear least squares:

```
γ(h) = nugget + sill × [ 1.5(h/a) − 0.5(h/a)³ ]   for h ≤ a
γ(h) = nugget + sill                                  for h > a
```

The **range parameter a** is the autocorrelation radius — the distance beyond which
knowing one tile's erosion count tells you nothing about a neighbour's count.

### Result: range = 31 m

The semi-variance curve flattens at **~31 m**. Erosion patches in this dataset have a
characteristic diameter of ~31 m. Within that radius, tiles are spatially correlated;
beyond it, they are independent.

---

## 6. Tile Size Derivation

### Current tile footprint

| Statistic | Value |
|---|---|
| Ground Sampling Distance (median) | **2.1 cm/px** (UAV at ~70 m altitude) |
| Standard tile size | 384 × 384 px |
| Tile footprint | 384 × 0.021 m = **8.1 m per side** |

### The mismatch

```
Autocorrelation range  :  31 m
Current tile footprint :   8 m
Ratio                  :  31 / 8 ≈ 3.8×
```

The current tile sees only **26% of a typical erosion patch diameter**. Most
erosion-containing tiles capture only a fragment of the patch they belong to.
These are precisely the tiles where recall collapses.

### Recommended tile size

To give the model a field of view covering at least one full erosion patch:

```
Required pixels = 31 m / 0.021 m·px⁻¹ = 1,476 px

Round up to the nearest multiple of 384:
  → 1,536 px  (= 4 × 384)

Verification:
  1,536 px × 0.021 m/px = 32.3 m  ≥  31 m  ✓
```

**Recommended tile size: 1,536 × 1,536 px**

This holds across the full GSD distribution of the dataset:

| GSD (cm/px) | Pixels for 31 m | Nearest candidate |
|---|---|---|
| 1.8 (finest) | 1,722 px | 1,792 px (≈ 4.67 × 384) |
| 2.1 (median) | 1,476 px | **1,536 px (= 4 × 384)** |
| 2.5 (coarse) | 1,240 px | 1,280 px (= 3.3 × 384) |

In all cases the answer falls in the **1,280–1,792 px** range. A single target of
**1,536 px** is robust across the full dataset.

---

## 7. Conclusions

| Metric | Overall | Key observation |
|---|---|---|
| Precision (erosion) | **94.7%** | High across all meaningful tiles; already near ceiling |
| Recall (erosion) | **86.9%** | Grows with tile density — context-sensitive recognition |
| F1 (erosion) | **90.6%** | Recall is the sole binding constraint |
| Lorenz Gini | **≈ 0.7** | Erosion is spatially concentrated in a few dense tiles |
| Autocorrelation range | **31 m** | Erosion patch diameter |
| Current tile footprint | **8.1 m** | Only 26% of patch diameter |
| Recommended tile | **1,536 px** | Full patch coverage at median GSD |

**The model is not fundamentally wrong — it is operating on tiles that are too small.**
It reliably identifies erosion when a coherent patch is visible in the receptive field.
When the tile shows only a thin fringe of erosion, the spatial cues are absent and recall
collapses. Increasing the tile size to 1,536 × 1,536 px is the highest-leverage
intervention before any architectural change, as it directly addresses the root cause
identified by the variogram analysis.
