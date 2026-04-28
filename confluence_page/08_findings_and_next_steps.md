# 8. Findings, Caveats & Next Steps

## 8.1 Key Findings

| Finding | Evidence | Implication |
|---|---|---|
| MiT-B3 generalizes best | Test F1=0.630 (+22% vs baseline) | Default architecture for next runs |
| v3 overfits catastrophically past ~400 epochs | Test F1 collapses from 0.606 (ep399) to 0.135 (ep1819) | Hard early stopping around ep400 for this dataset size |
| v2 dominance is an artefact | Data leakage: test zones seen during training | v2 test score (0.697) is not a valid reference |
| Recall is the binding constraint | Best model: Prec=0.759, Rec=0.538 | Precision is near-ceiling; improvement = recall improvement |
| Tile size is the structural bottleneck | Patch diameter 30.9 m vs tile footprint 8.1 m (26%) | 1,536 × 1,536 px tiles = highest-leverage next intervention |
| AMP + compile ≈ −40% wall time | ~1.45–1.55× speedup on Tensor Cores | Use by default for all future runs |
| Lorenz analysis: global F1 misleads | Gini 0.53, bottom 50% tiles → 13% of erosion | Always look at per-tile distributions, not just aggregate F1 |

---

## 8.2 Caveat: Task Metric ≠ Fine-Tuning Quality

A high F1 score on erosion measures exactly one thing: how well a model detects erosion on this specific dataset. It does not tell us whether that model is a good foundation for fine-tuning on a different task — for instance, the 6-class vegetation schema used in the TytonAI mega model.

A model trained to specialize in erosion texture + DSM features compresses its internal representations toward those cues. This may or may not transfer. The inverse is equally possible: a model that generalizes poorly on erosion may be a better fine-tuning foundation because it retained more general visual representations.

**This assumption is worth verifying empirically — not asserting.** The most direct validation path is A/B deployment: give one subset of TytonAI annotators the fine-tuned model and another subset the baseline, and compare annotation quality or coverage speed. This is connectable to MLflow tracking.

---

## 8.3 Connection to the Broader Fine-Tuning Question

This project established a rigorous methodology that can be directly reused:
- **Geographic train/test split** with proper leakage controls
- **Per-tile metric evaluation** with parquet output
- **Lorenz + variogram analysis** for characterizing data distribution
- **Model comparison dashboard** for benchmarking

These tools are directly applicable to the question of **mega model value**: does pre-training on a large multi-site corpus accelerate convergence or improve final performance on a new erosion site, compared to training from scratch with the same data?

The architecture comparison done here (UNet vs SegFormer, shared encoder vs native) provides useful context, but **we do not conclude from erosion F1 scores which architecture is best for fine-tuning in a different context**. That requires dedicated experiments — a reminder that results from a specialized task do not automatically generalize to the transfer learning regime.

---

## 8.4 Next Steps

| Priority | Action | Status |
|---|---|---|
| High | AMP + torch.compile training variants (9 runs: 3 architectures × 3 configs) | In progress |
| High | Retrain MiT-B3 at 1,536 × 1,536 px tile size | Not started |
| Medium | Consolidate training into TytonAI exclusively | Planned |
| Medium | Fine-tuning vs. from-scratch benchmark — dedicated experiment | Not started |
| Low | Extend test set to additional geographic zones | Opportunistic |
