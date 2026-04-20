# Lorenz Analysis — Interpretation & Spatial Autocorrelation

*Model: v3_split_test — epoch 1819 | 19,752 tiles total | 6,852 tiles with erosion*

---

## 1. Lorenz Curve — The Erosion Inequality is Extreme

50% of erosion-containing tiles hold only **13% of erosion pixels**.  
Inversely, the top ~10% most erosion-rich tiles hold the large majority of all erosion area.

This means most erosion tiles are **"trace" tiles**: they appear at the spatial boundary of an
erosion field, with only a handful of erosion pixels leaking into the frame. True erosion cores
are concentrated in a small fraction of the dataset.

**Practical implication:** dataset metrics and loss functions are dominated by these dense tiles.
Optimising for the tail (the boundary tiles) is both harder and less impactful on aggregate scores.

---

## 2. Precision — Fast Rise, then Plateau at 94.7%

The curve spikes from near 0% to ~80% in the first **5% of tiles** (i.e. the tiles with the
fewest erosion pixels), then climbs slowly to 94.7%.

### Why the initial spike looks alarming but isn't

Those first tiles have **1–10 actual erosion pixels**. For such a tile, a single false-positive
prediction (1 wrongly classified pixel in a sea of 147 k non-erosion pixels) collapses the
cumulative precision — numerically unstable near zero counts.

Crucially, the Lorenz curve tells us those tiles collectively hold **< 0.3% of all erosion**.
Their precision instability is a statistical artefact of near-zero denominators, not a real
model weakness.

### What this actually says

Once past that threshold, precision is already **85–90% and climbs smoothly to 94.7%**.  
For any tile where erosion is meaningfully present, what the model calls erosion is almost
always erosion. **The model is very discriminative; it rarely invents erosion out of nothing.**

---

## 3. Recall — Strictly Increasing, Near-Linear

Recall was **not** guaranteed to increase. It would have gone *down* if the most erosion-rich
tiles were the hardest (model overwhelmed by large patches → many false negatives dominating
the cumulative count). The fact that it is **strictly increasing** is a non-trivial result.

### Mechanism 1 — Detection threshold effect

On a tile with 5 erosion pixels scattered at a field boundary, the model's activation signal
may never cross the classification threshold. More erosion = stronger, more spatially coherent
activation = more pixels recalled.

### Mechanism 2 — Texture vs. structure

Erosion is recognisable as a *spatial pattern* (bare soil texture, slope context, connected
patches) rather than a pixel-by-pixel feature. When there is enough erosion to form that
structure in the receptive field, the model sees it. When it is just a fringe of 5 pixels, the
context is ambiguous.

### The near-linear shape

The marginal recall gain per added tile is roughly constant across the sorted range. This means
the detection improvement is smooth and predictable — no cliff, no saturation — which suggests
the model has not yet found a hard upper bound on recall.

---

## 4. Hypothesis — Bigger Tiles Could Help Recall

The recall curve is the strongest evidence for this. The core argument:

- Erosion is likely **spatially autocorrelated** (patches, not isolated pixels).
- Erosion is likely **context-dependent** to recognise (you need to see surrounding bare land,
  slope, texture, not just a local 3×3 kernel).
- A tile that straddles an erosion boundary gives the model a confusing context: half bare,
  half vegetated, little coherent erosion structure.
- A larger tile (or a tile centred within an erosion field) gives the model the full pattern.

The bottom ~30% of tiles by erosion count are probably exactly those boundary / fringe tiles,
and they are where recall collapses.

**Testable prediction:** if the spatial autocorrelation radius of erosion is larger than the
current tile footprint, increasing tile size should improve recall disproportionately on the
currently-weak tiles.

---

## 5. Variogram — How It Is Computed

### What the variogram measures

For every pair of tiles (i, j) separated by distance h, it measures how *different* their
erosion pixel counts are:

```
γ(h) = 0.5 × mean( (z_i − z_j)² )   for all pairs at distance h
```

- When h is **small** (nearby tiles), z_i ≈ z_j → (z_i − z_j)² ≈ 0 → γ(h) is low.
- As h grows, tiles become more and more dissimilar → γ(h) rises.
- Beyond the **range**, tiles are statistically independent: γ(h) plateaus at the **sill**.

### How the average is computed across sites

Tiles are first grouped into 48 geographic sites (DBSCAN, eps = 200 m). For each site,
all pairs within `max_dist_m` are enumerated and binned by distance. The final γ(h) for
each lag bin is a **weighted average across sites**, weighted by the number of pairs:

```
γ_total(h) = Σ_sites [ n_pairs_site(h) × γ_site(h) ]
           / Σ_sites [ n_pairs_site(h) ]
```

Sites with many tiles contribute more to the estimate; a site with 3 tiles barely moves the
needle against one with 200. This is numerically equivalent to pooling all within-site pairs
together and computing the global semi-variance.

With 30 valid lag bins each holding **40 k – 120 k pairs**, the estimate is very stable.

### What "range = 31 m" means

The semi-variance curve flattens at ~31 m. Beyond that distance, knowing one tile's erosion
pixel count tells you **nothing** about a neighbouring tile's count — they are independent.

Conversely, within 31 m, tiles are correlated: if tile A has a lot of erosion, tile B
at 10 m away is statistically likely to also have erosion. That correlation is the erosion
*patch* — it has a characteristic diameter of ~31 m.

---

## 6. Tile Resolution & Suggested Tile Size

### Current tile resolution

| Statistic | Value |
|---|---|
| Ground Sampling Distance (GSD) | 1.8 – 3.1 cm/px · **median ≈ 2.1 cm/px** |
| Standard tile size | **384 × 384 px** (80% of tiles) |
| Footprint (384 px × 2.1 cm/px) | **≈ 8.1 m per side** |

The GSD (~2.1 cm/px) is consistent with UAV/drone imagery at ~60–80 m flight altitude.
The tile footprint spans roughly **8 m × 8 m** on the ground.

### The mismatch: patch size vs. tile size

```
Autocorrelation range (patch diameter) :  31 m
Current tile footprint                 :   8 m
Ratio                                  :  31 / 8 ≈ 3.8×
```

The current tile sees only **~26% of a typical erosion patch diameter**. Most tiles that
contain any erosion at all are fringe tiles — they lie at the edge of a 31 m patch and
capture only a sliver of it. That is exactly the population where recall collapses.

### Calculation — recommended tile size

To give the model a field of view that contains at least one full erosion patch:

```
Target footprint  ≥ range = 31 m

At median GSD (2.1 cm/px):
  required pixels = 31 m / 0.021 m·px⁻¹ = 1 476 px

Round up to next power-of-2-friendly value:
  → 1 536 px  (= 4 × 384)

Verification:
  1 536 px × 0.021 m/px = 32.3 m  ✓  (≥ 31 m range)
```

**Recommended tile size: 1 536 × 1 536 px**, i.e. 4× the current tile side.

Note: at the GSD extremes the answer shifts slightly —

| GSD (cm/px) | Pixels needed for 31 m | Nearest tile candidate |
|---|---|---|
| 1.8 (finest) | 1 722 px | 1 792 px (= 4.67 × 384) |
| 2.1 (median) | 1 476 px | **1 536 px (= 4 × 384)** |
| 2.5 (coarse) | 1 240 px | 1 280 px (= 3.3 × 384) |

In all cases the answer is in the **1 280–1 792 px** range. A single target of **1 536 px**
is a robust choice across the full GSD distribution of this dataset.

---

## 7. Summary Table

| Metric | Overall value | Key observation |
|---|---|---|
| Lorenz Gini index | high (≈ 0.7) | Erosion is spatially concentrated |
| Precision (erosion) | 94.7% | High across all meaningful tiles |
| Recall (erosion) | 86.9% | Grows with tile erosion density — context-sensitive |
| F1 (erosion) | 90.6% | Recall is the binding constraint |

**The binding constraint is recall, not precision.**  
Improving recall on low-erosion (boundary) tiles is the highest-leverage intervention,
and tile size / tiling strategy is the most plausible lever before any model change.
