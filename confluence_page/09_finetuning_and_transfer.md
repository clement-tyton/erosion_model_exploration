# 9. Fine-Tuning Across Sites — Architecture Choice & the Fractal Nature of Erosion

## 9.1 The Transfer Learning Question

The erosion model trained here (MiT-B3, 19,752 tiles, Dampier/Marshall/Dugong test split) is a strong candidate as a fine-tuning foundation for a new erosion site. The question is: **which architecture transfers best when the site changes, the resolution changes, and the annotation quality changes?**

This is a different question from "which architecture has the best erosion F1" — and the answer depends on understanding what erosion actually looks like geometrically.

---

## 9.2 Erosion is Fractal

Erosion patterns exhibit **self-similarity across spatial scales** — the defining property of fractal geometry. A gully system observed at 50 cm/px shows the same branching dendritic network structure as a rill system observed at 2 cm/px. The V-shaped cross-sections, the hierarchical branching, the preferential flow paths along slopes — these structural patterns repeat at every scale of observation.

This has two direct consequences for any machine learning model trained on erosion imagery:

**1. Changing resolution is not a zoom — it accesses a different level of the fractal hierarchy.**

At 2.1 cm/px (current dataset, UAV at ~70 m altitude), the model sees micro-scale features: small rills, surface crusting, the precise edge between bare soil and vegetation cover. At 10 cm/px (lower-altitude UAV or lower-quality flight), those micro-features merge into mesoscale gully structures. At 50 cm/px (satellite), you see the macro-scale drainage network.

These are not the same image at a different zoom level. They are genuinely different phenomena along the same fractal cascade. A model trained to recognize erosion texture at 2.1 cm/px is not looking at a "blurred version" of the same thing at 10 cm/px — it is looking at different structural entities entirely.

**2. Erosion boundaries are inherently ambiguous — annotation quality variation is a fractal boundary problem.**

At any finite resolution, there is no geometrically "correct" erosion boundary. The edge between eroded soil and stable ground is itself fractal: zoom in further and you find more boundary, not a clean line. Different annotators resolve this ambiguity differently. A model that memorizes exact pixel-level boundary positions from a specific annotation session has overfit to one arbitrary resolution of a fractal boundary — not to the underlying physical phenomenon.

---

## 9.3 Why This Matters for Architecture Choice

The two architectures behave very differently under these conditions:

### CNN Encoder (UNet + res2net) — texture-bound, scale-specific

CNN kernels operate at a fixed spatial scale. A 3×3 kernel at stride 1 encodes a 3×3 pixel neighborhood — it learns "what does erosion look like at this exact pixel pitch." This is simultaneously the encoder's strength (very precise at the training resolution) and its fundamental limitation:

- **Site change:** The model memorized local texture statistics (soil color, shadow pattern, vegetation boundary appearance) specific to the training captures. A new site with different soil type, different vegetation density, or different UAV sensor will present different pixel-level statistics for the same physical erosion. The CNN feature maps shift out of distribution.

- **Resolution change:** At a different GSD, the same physical erosion patch occupies a different number of pixels. The features the CNN learned (optimized for N×N pixel patches at 2.1 cm/px) no longer match the new input. The model is not seeing a different zoom of the same representation — it is seeing an input its kernels were never exposed to. Fine-tuning effectively needs to re-learn the feature extraction from scratch.

- **Annotation ambiguity:** The UNet encoder, trained with LR=1e-3 for 400 epochs, overfits to train F1 = 0.96. It has memorized the exact boundary decisions of the original annotation session. Differently-annotated data presents conflicting supervision at the pixel level.

### MiT-B3 (SegFormer) — structure-bound, scale-adaptive

The Mix Transformer encoder processes the image through **hierarchical self-attention**. Each token attends to all other tokens — there is no fixed spatial kernel. The model learns which spatial relationships matter, not which pixel neighborhood configurations matter.

This maps onto fractal erosion geometry in a structurally deep way:

- **Site change:** Structural features of erosion — the shape and connectivity of eroded patches, their relationship to slope direction, the way bare soil grades into vegetated boundary — are **geographically invariant**. A dendritic erosion network in Western Australia and one in East Africa share the same topological structure. Attention captures relationships between regions, not the pixel values of those regions.

- **Resolution change:** Self-attention is not pixel-scale-bound. At a different resolution, the same erosion patch covers a different number of tokens — but the spatial relationships between those tokens (this region is eroded, that region is stable, they share this boundary configuration) remain informative. The model has to re-calibrate its scale expectations, but it does not need to re-learn what erosion *is*.

- **Fractal multi-scale structure:** The MiT architecture uses 4 hierarchical stages with progressively coarser resolution feature maps (H/4, H/8, H/16, H/32). This hierarchy naturally captures erosion features at multiple scales simultaneously — micro-texture at early stages, patch-level shape at intermediate stages, global slope context at deep stages. Fractal patterns are multi-scale by definition. A multi-scale encoder is a better match.

- **Annotation robustness:** The model trained with encoder LR=6e-6 barely updated the pre-trained representations. It learned to apply general segmentation features (from ADE20K) to erosion structure, without memorizing specific boundary positions. Differently-annotated data with the same physical erosion will be close to in-distribution for the encoder.

---

## 9.4 Practical Fine-Tuning Strategy

Given the above, the recommended approach for a new erosion site with different resolution and/or annotation quality:

### Option A — Encoder frozen, head only (few-shot new site)

```
Freeze MiT-B3 encoder entirely
Train only the SegFormer decode head
LR: 1e-4 to 1e-3 on the head
```

Use this when: the new site has few annotated tiles (< 2,000), or annotation quality is uncertain. The frozen encoder provides stable structural features. The head re-calibrates the decision boundary for the new domain without risk of overfitting. Convergence is fast — 20–50 epochs is typically sufficient.

### Option B — Differential fine-tuning (full new-site adaptation)

```
Encoder LR: 1e-6 to 6e-6   (same as original training or lower)
Head LR:    6e-5 to 1e-4
Gradient clipping: 0.5
Warmup: 5–10% of epochs
Early stopping based on a held-out geographic zone from the new site
```

Use this when: the new site is very different (large resolution gap, very different biome), and enough annotated data exists to support full fine-tuning. The extremely low encoder LR ensures the structural representations are preserved rather than overwritten.

### What NOT to do

Do not fine-tune with a uniform high LR (1e-3). As the MLflow data confirmed, this destroys the pre-trained encoder in the early epochs and test F1 collapses to zero. The overfitting behavior documented for UNet at 400 epochs will reproduce within 50 epochs under these conditions.

---

## 9.5 The One Empirical Test That Would Confirm This

The argument above is theoretical and consistent with the literature, but it has not been validated on this specific dataset family. The direct test:

1. Hold out a geographically isolated zone from the new site as a validation set
2. Fine-tune MiT-B3 (from erosion weights) — Option A or B above
3. Fine-tune UNet (from erosion weights) — matched epochs and early stopping
4. Compare validation F1 at convergence AND at a small annotation budget (e.g. 500 tiles)

The small-annotation-budget comparison is the most diagnostic: if MiT-B3 needs 500 tiles to reach the same F1 that UNet needs 2,000 tiles to reach, the structural transfer hypothesis is validated.

---

## 9.6 Summary

| Factor | MiT-B3 | UNet / res2net |
|---|---|---|
| Site change | Structural features transfer across geography | Texture features are site-specific |
| Resolution change | Attention is not pixel-scale-bound | CNN kernels are trained at a fixed pixel pitch |
| Annotation quality variation | Robust — did not memorize boundaries | Fragile — overfit to original annotation style |
| Fractal multi-scale structure | Natural match — hierarchical attention | Limited — fixed kernel scale per stage |
| Fine-tuning controllability | High — freeze encoder, tune head | Low — uniform LR, fast overfitting |
| Few-shot new site | Viable — head-only fine-tuning works | Risky — no equivalent control mechanism |

**Recommendation: fine-tune from MiT-B3 erosion weights**, using Option A (frozen encoder) for small annotation budgets or high annotation uncertainty, Option B (differential LR) for large new-site datasets. The fractal nature of erosion geometry is the structural reason why attention-based representations generalize better across the resolution and site changes that characterize real-world deployment conditions.
