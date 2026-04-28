# 5. Model Lineage

## Family Tree

```
v1 — Jaswinder baseline (ep50)
  └─ v2 — false-positive fix (ep50)  ⚠️ no geographic split
       └─ v3 — geographic split (ep50 → ep1998)
            └─ v4 — TytonAI fine-tune (ep5, incomplete)

MLflow — independent, same 19,752-tile dataset, proper split:
  ├─ mlflow_unet_baseline (ep200)
  ├─ mlflow_segf_res2net (ep200)
  └─ mlflow_segf_mit_b3 (ep200)  ← current best
```

---

## v1 — Jaswinder Baseline

**Found on arrival.** Trained on 27,440 tiles (a larger dataset than what became v3), 50 epochs via TytonAI ObjectTrain.

- Test F1: **0.259** (Prec=0.652, Rec=0.162)
- High precision but catastrophic recall — the model misses most erosion
- No geographic test split; numbers are not directly comparable

The dominant failure mode: the model only fires on dense, unambiguous erosion patches. Any fringe or partial patch is ignored.

---

## v2 — False Positive Fix

**Context:** after observing false positives in the field with Keith, v2 was built by fine-tuning v1 with added no-erosion training data, pushing false positive rate down. 50 epochs on 21,213 tiles.

- Test F1: **0.697** (Prec=0.783, Rec=0.628)

⚠️ **This score is not a valid generalization measure.** v2 was trained on all available data **including captures from the Dampier, Marshall, and Dugong zones** that later became the test set. There is direct data leakage between train and test. The high score reflects memorization, not generalization.

v2 remains a useful reference to understand what "all data" performance looks like before a proper split — and as a reminder that such numbers can be misleading.

---

## v3 — Proper Geographic Split

**The first model trained with a geographically isolated test set.** Retrained from scratch on 19,752 balanced tiles, with Dampier/Marshall/Dugong strictly excluded.

Multiple checkpoints were evaluated:

| Checkpoint | Test Prec | Test Rec | Test F1 | Note |
|---|---|---|---|---|
| ep50 | 0.677 | 0.419 | **0.517** | **TytonAI baseline** |
| ep78 | 0.833 | 0.423 | 0.561 | |
| ep80 | 0.855 | 0.408 | 0.552 | |
| ep95 | 0.265 | 0.573 | 0.362 | Divergence |
| ep243 | 0.111 | 0.510 | 0.182 | Diverged |
| ep399 | 0.776 | 0.497 | **0.606** | Best honest v3 |
| ep400 | 0.538 | 0.278 | 0.366 | Spike at exact ep400 |
| ep1819 | 0.083 | 0.359 | **0.135** | Catastrophic overfit |
| ep1998 | 0.114 | 0.371 | 0.175 | Still overfit |

**Key observation — catastrophic overfitting:** v3 ep1819 is the best model on the training set (F1=0.906, Prec=0.946, Rec=0.869). On the test set, it collapses to F1=0.135. The model has memorized the training distribution and lost all generalization. This defines the upper bound on training duration for this dataset size: **early stopping around epoch 399 is critical.**

The divergence at ep95 and ep243 is likely a learning rate schedule artifact; the model recovers before ep399.

`[IMAGE PLACEHOLDER: Train F1 vs Test F1 curve — shows v3 train climbing to 0.906 while test peaks at 0.606 (ep399) then collapses. Source: MLflow metrics or dashboard Compare tab with v3 checkpoints selected.]`

---

## v4 — TytonAI Fine-Tune

v3 fine-tuned for 5 epochs on TytonAI platform data. Only ep5 was evaluated:
- Test F1: **0.017** (Prec=0.527, Rec=0.008)

Barely started — not representative. Included in the registry for completeness.

---

## MLflow Series — Architecture Benchmark

Trained independently using the same 19,752-tile dataset and the same geographic split as v3. Same hyperparameters as TytonAI ObjectTrain. 200 epochs tracked via MLflow.

### mlflow_unet_baseline (ep200)
Architecture: UNet + res2net101 encoder.
- Train: Prec=0.591, Rec=0.241, F1=0.342
- Test: Prec=0.545, Rec=0.207, F1=0.300

Underperforms v3_ep50 on test. The UNet with res2net encoder reaches a ceiling early on this data.

### mlflow_segf_res2net (ep200)
Architecture: SegFormer + res2net101 encoder (same encoder as UNet, different decoder).
- Train: Prec=0.733, Rec=0.339, F1=0.463
- Test: Prec=0.755, Rec=0.241, F1=0.365

Better than UNet on train, similar on test. The shared encoder limits recall in both cases.

### mlflow_segf_mit_b3 (ep200) — Current Best

Architecture: SegFormer with its native MiT-B3 encoder (Mix Transformer, pre-trained on ImageNet + ADE20K semantic segmentation).
- Train: Prec=0.738, Rec=0.799, F1=**0.767**
- Test: Prec=0.759, Rec=0.538, F1=**0.630**

**This is the current best model.** It achieves +22% test F1 over the v3_ep50 TytonAI baseline (+0.113 absolute). The key differentiator is recall — the MiT-B3 encoder's attention mechanism captures patch-level spatial context that CNN-based encoders miss at this tile size.

Training with MiT-B3 is less stable (requires warmup, lower LR, tighter gradient clipping), but the performance gain is consistent.

`[IMAGE PLACEHOLDER: Training curves for MiT-B3 — loss, iou_erosion, f1_erosion across 200 epochs. Stable convergence after warmup. Source: MLflow "Erosion project" > segformer_mit_b3_stable run.]`
