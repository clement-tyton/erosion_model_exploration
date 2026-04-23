#!/usr/bin/env bash
# Replicate 3 baseline runs × 3 configs (AMP / compile / AMP+compile)
# All other hyperparams identical to the original runs.
# Run from repo root: bash RUN_AMP_COMPILE.sh

set -e

# ── AMP only ──────────────────────────────────────────────────────────────────
python Experiments_MLFLOW/run_experiment.py --arch unet \
    --run-name unet_baseline_amp --amp

python Experiments_MLFLOW/run_experiment.py --arch segformer \
    --encoder timm-res2net101_26w_4s \
    --run-name segf_amp --amp

python Experiments_MLFLOW/run_experiment.py --arch segformer \
    --encoder mit_b3 \
    --run-name segformer_mit_b3_amp --amp

# ── torch.compile only (max-autotune, cudagraphs disabled for grad accumulation) ─
python Experiments_MLFLOW/run_experiment.py --arch unet \
    --run-name unet_baseline_compile --compile-mode max-autotune-no-cudagraphs

python Experiments_MLFLOW/run_experiment.py --arch segformer \
    --encoder timm-res2net101_26w_4s \
    --run-name segf_compile --compile-mode max-autotune-no-cudagraphs

python Experiments_MLFLOW/run_experiment.py --arch segformer \
    --encoder mit_b3 \
    --run-name segformer_mit_b3_compile --compile-mode max-autotune-no-cudagraphs

# ── AMP + torch.compile ───────────────────────────────────────────────────────
python Experiments_MLFLOW/run_experiment.py --arch unet \
    --run-name unet_baseline_amp_compile --amp --compile-mode max-autotune-no-cudagraphs

python Experiments_MLFLOW/run_experiment.py --arch segformer \
    --encoder timm-res2net101_26w_4s \
    --run-name segf_amp_compile --amp --compile-mode max-autotune-no-cudagraphs

python Experiments_MLFLOW/run_experiment.py --arch segformer \
    --encoder mit_b3 \
    --run-name segformer_mit_b3_amp_compile --amp --compile-mode max-autotune-no-cudagraphs
