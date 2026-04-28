# 2. Dataset Design & Geographic Split

## 2.1 Training Set

Tiles are extracted from TytonAI as NPZ archives (one imagery file + one mask file per tile). A `balanced_tiles.json` manifest (19,752 entries) drives the dataloader. Each entry contains:

```json
{
  "imagery_file": "image_<uuid>_1536_1536.npz",
  "mask_file":    "mask_<uuid>_1536_1536.npz",
  "count":        14,
  "bands":        ["MEP", "RED", "GREEN", "BLUE", "DSM", "DSM_NORMALIZED"]
}
```

| Statistic | Value |
|---|---|
| Total tiles | 19,752 |
| Tiles with erosion (≥1 px) | 6,852 (34.7%) |
| Total erosion pixels | 98,468,414 |
| Median tile footprint | 8.1 m × 8.1 m |
| Median GSD | ~2.1 cm/px (UAV at ~70 m altitude) |
| Geographic sites (DBSCAN, ε=200 m) | 48 |

The tile manifest was obtained from the **input** of the TytonAI ObjectTrain workflow (tile UUID `ef1410ef-59ab-4821-b044-11f8ef6a040a`). Band statistics (mean, std, class weights) were extracted from `DATABALANCE_CONFIG.JSON`:

- `TRAIN_MEAN = [RED_mean, GREEN_mean, BLUE_mean, DSM_NORM_mean]`
- `TRAIN_STD  = [RED_std, GREEN_std, BLUE_std, DSM_NORM_std]`
- `CLASS_WEIGHTS = [1.0, 1.764]`  (erosion class upweighted in CrossEntropy loss)

---

## 2.2 Test Set — Strict Geographic Isolation

Three capture zones were held out entirely from training. No tile from these zones appears in any training set from v3 onward.

| Zone | Capture name | Tiles | Note |
|---|---|---|---|
| Dampier | 230826_Dampier_1-3 | 771 | |
| Marshall | 230606_Marshall_13 | 512 | |
| Dugong | 230510_Dugong_1 | 483 | |
| **Total** | | **1,766** | 637 with erosion (36.1%) |

This split was introduced for v3. **v1 and v2 were trained before this split existed — they saw captures from these zones during training, which invalidates their test scores as generalization measures.**

`[IMAGE PLACEHOLDER: Map — 48 training site clusters (colored by DBSCAN site ID) + 3 test zones (Dampier, Marshall, Dugong) highlighted. Source: dashboard Map tab or Excalidraw.]`

---

## 2.3 What "Balanced" Means

The manifest was produced by TytonAI's ObjectTrain balancing logic, which ensures that erosion-heavy and erosion-light tiles are sampled in controlled proportions. This prevents the training loss from being dominated by the ~65% no-erosion tiles. The class weight `1.764` on the erosion class in the CrossEntropy loss further compensates for the imbalance at the pixel level.
