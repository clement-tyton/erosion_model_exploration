# 6. Model Comparison & Results

## 6.1 Training Set — Global F1 (micro, pixel-level)

Evaluated on the same 19,752-tile training set used by v3/MLflow models. v1 and v2 used different datasets (see model lineage).

| Model | Dataset tiles | Prec | Rec | F1 | IOU |
|---|---|---|---|---|---|
| v1 jaswinder ep50 | 27,440 | 0.578 | 0.104 | 0.176 | 0.096 |
| v2 no-erosion ep50 ⚠️ | 21,213 | 0.744 | 0.599 | 0.664 | 0.497 |
| v3 ep50 — **baseline** | 19,752 | — | — | — | — |
| v3 ep399 | 19,752 | — | — | — | — |
| v3 ep1819 | 19,752 | 0.946 | 0.869 | 0.906 | 0.827 |
| mlflow UNet ep200 | 19,752 | 0.591 | 0.241 | 0.342 | 0.206 |
| mlflow SegF res2net ep200 | 19,752 | 0.733 | 0.339 | 0.463 | 0.301 |
| **mlflow SegF MiT-B3 ep200** | **19,752** | **0.738** | **0.799** | **0.767** | **0.622** |

*v3 ep50 and ep399 train-set metrics not stored separately — they are intermediate checkpoints of the same run.*

⚠️ v2 trained on all data including future test zones. Train score is not comparable.

---

## 6.2 Test Set — Dampier + Marshall + Dugong (1,766 tiles, 637 with erosion)

| Model | Prec | Rec | F1 | Δ vs baseline | Note |
|---|---|---|---|---|---|
| v1 jaswinder ep50 | 0.652 | 0.162 | 0.259 | −0.258 | |
| v2 no-erosion ep50 | 0.783 | 0.628 | 0.697 | — | ⚠️ Data leakage |
| **v3 ep50 — baseline** | **0.677** | **0.419** | **0.517** | **0** | First valid split |
| v3 ep78 | 0.833 | 0.423 | 0.561 | +0.044 | |
| v3 ep80 | 0.855 | 0.408 | 0.552 | +0.035 | |
| v3 ep399 | 0.776 | 0.497 | 0.606 | **+0.089** | Best honest v3 |
| v3 ep1819 | 0.083 | 0.359 | 0.135 | −0.382 | Overfit |
| mlflow UNet ep200 | 0.545 | 0.207 | 0.300 | −0.217 | |
| mlflow SegF res2net ep200 | 0.755 | 0.241 | 0.365 | −0.152 | |
| **mlflow SegF MiT-B3 ep200** | **0.759** | **0.538** | **0.630** | **+0.113** | **Best** |

`[IMAGE PLACEHOLDER: Bar chart — test F1 for all models sorted descending. MiT-B3 and v3_ep399 highlighted. v2 shown with warning annotation. Source: dashboard Compare Models tab, test F1 column.]`

---

## 6.2b MLflow Training Curves — What Really Happened

Querying the MLflow server directly reveals a different picture than the epoch-200 checkpoint metrics suggest.

| Run | LR (encoder) | Train F1 (ep400) | Test F1 (ep400) | Diagnosis |
|---|---|---|---|---|
| `unet_baseline` | 1e-3 | 0.9637 | **0.1706** | Catastrophic overfit — mirrors v3 |
| `segf` (res2net) | 1e-3 | 0.9678 | **0.3000** | Catastrophic overfit |
| `segformer_mit_b3_stable` (wrong LR) | 1e-3 | 0.8878 | **0.0000** | Training diverged completely |
| `segformer_mit_b3_stable` (correct LR) | 6e-6 | 0.8439 | **0.6268** | Genuine generalization |

**Both UNet and SegFormer-res2net overfit to train F1 ~0.96 — the same catastrophic pattern as v3.**

The "train" metrics shown in section 6.1 (e.g. UNet = 0.342) are computed by evaluating the epoch-200 checkpoint on `balanced_tiles.json`, which does not exactly match the tiles the MLflow models trained on (MLflow trained on 29,181 tiles). The parquet "train" score therefore measures cross-tile generalization within the training distribution — not in-sample memorization.

**MiT-B3 is the only architecture that avoids catastrophic overfitting**, and only when trained with the correct differential LR (6e-6 encoder / 6e-5 head). At 1e-3 (same as UNet), MiT-B3 diverges completely — the high LR destroys the pre-trained encoder weights in the first few epochs, collapsing test F1 to zero.

---

## 6.3 SegFormer vs. UNet

| Aspect | UNet + res2net | SegFormer + res2net | SegFormer + MiT-B3 |
|---|---|---|---|
| Train F1 (ep200) | 0.342 | 0.463 | 0.767 |
| Test F1 (ep200) | 0.300 | 0.365 | **0.630** |
| Training stability | High — converges smoothly | Medium | Lower — needs warmup, low LR, tight clip |
| Recall on test | 0.207 | 0.241 | **0.538** |
| Key strength | Reliable baseline | Marginal improvement | Global context capture |

**Why MiT-B3 wins on recall:** the MiT encoder uses multi-head self-attention, which gives the model a global receptive field. It can "see" the spatial structure of an erosion patch (bare soil texture + connected slope context) even when the patch extends beyond individual CNN kernel windows. At 384 px tile size, this matters — the model can integrate evidence across the full tile.

**Why res2net-based models plateau:** the res2net encoder was pre-trained on ImageNet only (classification features, not segmentation). The MiT-B3 encoder was additionally pre-trained on ADE20K semantic segmentation — it already "knows" how to produce spatially meaningful feature maps.

---

## 6.4 Overfitting Diagnosis — v3 Beyond ep400

`[IMAGE PLACEHOLDER: Train F1 vs Test F1 as a function of epoch (v3). Train climbs monotonically to 0.906 at ep1819. Test peaks at ep399 (0.606), then collapses to 0.135 by ep1819. Source: MLflow metrics for all v3 checkpoint evaluations.]`

The gap between train and test F1 is the classic overfitting signature. At 19,752 tiles, the model has enough capacity to memorize the training distribution within ~400 epochs. Beyond that, it loses the ability to generalize.

**Practical implication:** for this dataset size, **400 epochs is the approximate upper bound**. Any future training run should apply early stopping based on a validation set, or use the test score at checkpoint intervals as a guide.

---

## 6.5 Pairwise Comparison: MiT-B3 vs v3_ep399

These are the two best models on the test set. At the tile level:

`[IMAGE PLACEHOLDER: Dashboard pairwise deep-dive — scatter plot (x = v3_ep399 F1, y = MiT-B3 F1 per tile, colored by winner, sized by erosion pixel count) + Δ F1 histogram. Source: Compare Models tab → Pairwise Deep-Dive, select mlflow_segf_mit_b3 vs v3_ep399.]`

MiT-B3 wins on recall (+0.041 absolute), primarily on medium-density erosion tiles where the spatial context of the patch becomes visible. v3_ep399 has marginally higher precision (0.776 vs 0.759).
