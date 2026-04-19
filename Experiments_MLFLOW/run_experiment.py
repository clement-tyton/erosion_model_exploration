"""
Entry point for running UNet / SegFormer experiments.

Usage:
    python Experiments_MLFLOW/run_experiment.py --arch unet --epochs 400
    python Experiments_MLFLOW/run_experiment.py --arch segformer --variant b2 --epochs 400

    # Custom batch / accumulation
    python Experiments_MLFLOW/run_experiment.py --arch unet \\
        --batch-size 64 --accumulation-steps 4 --epochs 400

Effective batch size = batch-size × accumulation-steps (default 64 × 4 = 256).
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from torch.utils.data import DataLoader

# ── Make Experiments_MLFLOW importable when running as a script ───────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import Experiments_MLFLOW.config as cfg
from Experiments_MLFLOW.data.dataset import TrainDataset, TestDataset, collate_pad
from Experiments_MLFLOW.models import build_model
from Experiments_MLFLOW.training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train UNet or SegFormer on erosion dataset.")
    p.add_argument("--arch",    choices=["unet", "segformer"], required=True)
    p.add_argument("--encoder", default=cfg.ENCODER_NAME,
                   help=f"Encoder backbone for both archs (default: {cfg.ENCODER_NAME}). "
                        "Use 'mit_b2' / 'mit_b4' for native SegFormer MiT encoders.")
    p.add_argument("--epochs",  type=int, default=cfg.NUM_EPOCHS)
    p.add_argument("--batch-size",         type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--accumulation-steps", type=int, default=cfg.ACCUMULATION_STEPS)
    p.add_argument("--workers",            type=int, default=cfg.NUM_WORKERS)
    p.add_argument("--device", default=None,
                   help="Force device: 'cuda:0', 'cuda:1', 'cpu' (default: least-busy GPU)")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--eval-every",       type=int, default=10,
                   help="Run test evaluation every N epochs (default: 10)")
    p.add_argument("--run-name", default=None,
                   help="MLflow run name (default: arch_encoder_timestamp)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = args.device or os.environ.get("OBJECT_TRAIN_DEVICE") or cfg.select_free_gpu()
    print(f"Device: {device}")

    # ── Run name ──────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Short encoder label: "timm-res2net101_26w_4s" → "res2net101"
    enc_short = args.encoder.replace("timm-", "").split("_")[0]
    run_name = args.run_name or f"{args.arch}_{enc_short}_{timestamp}"
    print(f"Run name: {run_name}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading train dataset …")
    train_ds = TrainDataset(
        balanced_tiles_json=cfg.BALANCED_TILES_JSON,
        data_dir=cfg.TRAIN_DATA_DIR,
        model_bands=cfg.MODEL_BANDS,
        mean=cfg.TRAIN_MEAN,
        std=cfg.TRAIN_STD,
        train=True,
    )
    print(f"  {len(train_ds):,} train entries (with duplication)")

    print("Loading test dataset …")
    test_ds = TestDataset(
        test_metadata_json=cfg.TEST_METADATA_JSON,
        data_dir=cfg.TEST_DATA_DIR,
        model_bands=cfg.MODEL_BANDS,
        mean=cfg.TRAIN_MEAN,
        std=cfg.TRAIN_STD,
    )
    print(f"  {len(test_ds):,} test tiles")

    if len(train_ds) == 0:
        print("ERROR: No train tiles found. Run data/download_train.py first.")
        sys.exit(1)
    if len(test_ds) == 0:
        print("ERROR: No test tiles found. Run data/download_test.py first.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_pad,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_pad,
        pin_memory=(device == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Building model: {args.arch}  encoder: {args.encoder}")
    model = build_model(
        args.arch,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        encoder_name=args.encoder,
        encoder_depth=cfg.ENCODER_DEPTH,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    eff_bs = args.batch_size * args.accumulation_steps
    print(f"  Physical batch: {args.batch_size}  ×  accum: {args.accumulation_steps}  "
          f"→  effective: {eff_bs}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=cfg,
        run_name=run_name,
        arch=args.arch,
        encoder_name=args.encoder,
        device=device,
        accumulation_steps=args.accumulation_steps,
        checkpoint_every=args.checkpoint_every,
        eval_every=args.eval_every,
    )
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
