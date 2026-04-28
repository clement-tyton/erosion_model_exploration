# 4. Training Infrastructure

## 4.1 Two Parallel Pipelines

This project ran training on two parallel systems:

| | TytonAI ObjectTrain | MLflow (out-of-platform) |
|---|---|---|
| Platform | TytonAI | Local / GCP Cloud Run |
| Tracking | TytonAI workflow outputs (JSON) | MLflow server |
| Metadata source | ObjectTrain/TestEpoch workflow outputs | `balanced_tiles.json` (ObjectTrain input) |
| Models produced | v1, v2, v3 series, v4 fine-tune | unet_baseline, segf_res2net, segf_mit_b3 |
| Architecture code | ObjectTrain activity (internal) | `Experiments_MLFLOW/` (this repo) |

> **Note on this approach:** running training "à cheval" across two systems adds friction — metadata extraction, format adaptation, version tracking. This is explicitly the **last project** run this way. The medium-term ambition is to experiment exclusively within TytonAI, with the MLflow logic eventually absorbed into ObjectTrain.

### Mode Operandi

1. Run a training workflow on TytonAI
2. Extract metadata from the `TestEpoch` activity output (JSON) and the `ObjectTrain` input (balanced tiles JSON)
3. Download model weights from S3 (`epoch_file_key` from `GetModelEpoch` activity)
4. Apply model locally to train/test tiles → per-tile metrics parquet
5. Load metrics into the Streamlit dashboard

---

## 4.2 Training Hyperparameters

These are the hyperparameters used for all MLflow runs. The TytonAI runs use the same values, sourced from `OBJECT_TRAIN_INPUT_JSON` and `DATABALANCE_CONFIG.JSON`.

| Parameter | Value |
|---|---|
| Architecture | UNet or SegFormer (segmentation-models-pytorch) |
| Encoder — shared default | timm-res2net101_26w_4s |
| Encoder — SegFormer native | mit_b3 |
| Epochs | 400 |
| Physical batch size | 32 |
| Gradient accumulation steps | 8 → effective batch = **256** |
| Initial LR — UNet (Adam) | 1e-3 |
| Initial LR — SegFormer head | 6e-5 |
| Initial LR — SegFormer encoder | 6e-6 |
| LR schedule | Linear warmup (5% of epochs) → cosine decay |
| Gradient clipping | 1.0 (UNet/CNN), 0.5 (SegFormer) |
| Loss | CrossEntropy, class weights [1.0, 1.764] |
| Input channels | 4 (RED, GREEN, BLUE, DSM_NORMALIZED) |
| Classes | 2 (no-erosion, erosion) |

**Note on SegFormer LR:** SegFormer uses differential learning rates — the decode head (randomly initialized) trains at 6e-5 while the MiT encoder backbone (pre-trained on ImageNet + ADE20K) trains at 6e-6. Without this differentiation, the encoder pre-training is destroyed early in training.

---

## 4.3 Training Speed Optimizations

To reduce wall-clock time for 400-epoch runs, three configurations were added on top of the baseline:

| Configuration | Expected speedup | Mechanism |
|---|---|---|
| Baseline | 1× | float32, no compile |
| AMP (`--amp`) | ~1.35–1.45× | float16 forward/backward on Tensor Cores via `torch.autocast` + `GradScaler` |
| torch.compile only | ~1.15–1.3× | Triton kernel fusion via `torch.compile` |
| AMP + compile (`--amp --compile-mode max-autotune-no-cudagraphs`) | ~1.45–1.55× | Both combined |

**Why `max-autotune-no-cudagraphs` and not `max-autotune`:** CUDAGraphs capture and replay a static execution graph, which conflicts with gradient accumulation (the model forward is called N times before an optimizer step, overwriting buffers still needed by backward). The `no-cudagraphs` variant preserves Triton autotuning — the main performance gain — while avoiding this conflict.

For a 400-epoch MiT-B3 run (~347 s/epoch), AMP alone saves approximately **1.5–2 hours** of total training time.

`[IMAGE PLACEHOLDER: MLflow UI — training curves (loss, iou_erosion, f1_erosion) for baseline vs AMP vs compile runs, shown side-by-side. Source: MLflow experiment "Erosion project".]`

---

## 4.4 MLflow Tracking

All MLflow runs log:

- **Params:** architecture, encoder, epochs, batch size, accumulation steps, LR, class weights, AMP flag, compile mode
- **Metrics per epoch:** train loss, train IOU erosion, train F1, test IOU, test F1, test precision, test recall, epoch duration (minutes), gradient norm (mean/max)
- **Artifacts:** model checkpoints every 10 epochs

MLflow server: GCP Cloud Run (URI in `Experiments_MLFLOW/config.py`). Experiment name: `"Erosion project"`.
