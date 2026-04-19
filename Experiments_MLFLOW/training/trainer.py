"""
Training loop with MLflow integration and gradient accumulation.

Loss: CrossEntropyLoss(weight=class_weights, ignore_index=255)
      — matches production objecttrain (focal_loss=false).

Gradient accumulation:
    effective_batch = physical_batch × accumulation_steps
    loss divided by accumulation_steps before backward().
    optimizer.step() every accumulation_steps batches.

Metrics:
    Training  : ConfusionMeter  → global IOU/F1/precision/recall per epoch
    Test/Val  : TileMetricsCollector → per-tile parquet + global metrics

MLflow tracking: remote server from config.MLFLOW_TRACKING_URI
Parquet output : Experiments_MLFLOW/results/{run_name}/epoch_{N}_test_metrics.parquet
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import ConfusionMeter, TileMetricsCollector


def _git_tags() -> dict:
    def _run(cmd):
        try:
            return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    commit = _run(["git", "rev-parse", "HEAD"])
    if not commit:
        return {}
    return {
        "git.commit_short": _run(["git", "rev-parse", "--short", "HEAD"]),
        "git.branch":       _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git.dirty":        str(bool(_run(["git", "status", "--porcelain"]))),
    }


def _model_stats(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb   = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 ** 2
    return {
        "model/total_params":     round(total / 100_000) * 100_000,
        "model/trainable_params": round(trainable / 100_000) * 100_000,
        "model/size_mb":          round(size_mb, 1),
    }


class Trainer:
    """
    Parameters
    ----------
    model            : nn.Module
    train_loader     : DataLoader  (TrainDataset)
    test_loader      : DataLoader  (TestDataset)
    config           : Experiments_MLFLOW.config module
    run_name         : str  — MLflow run name
    arch             : str  — "unet" or "segformer"
    encoder_name     : str  — encoder backbone (logged to MLflow)
    device           : str  — "cuda:0", "cuda:1", "cpu", …
    accumulation_steps : int  (default 8 → effective batch = physical × 8)
    checkpoint_every   : int  (default 10 epochs)
    eval_every         : int  (default 10 epochs — run test pass + save parquet)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config,
        run_name: str,
        arch: str,
        encoder_name: str = "",
        device: str = "cuda",
        accumulation_steps: int = 4,
        checkpoint_every: int = 10,
        eval_every: int = 10,
    ):
        self.model         = model.to(device)
        self.train_loader  = train_loader
        self.test_loader   = test_loader
        self.config        = config
        self.run_name      = run_name
        self.arch          = arch
        self.encoder_name  = encoder_name or getattr(config, "ENCODER_NAME", "")
        self.device        = device
        self.accumulation_steps = accumulation_steps
        self.checkpoint_every   = checkpoint_every
        self.eval_every         = eval_every

        # ── Loss: CrossEntropyLoss — matches objecttrain (focal_loss=false) ──
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=config.IGNORE_INDEX,
        )

        # ── Optimizer + scheduler (Adam + StepLR — matches objecttrain) ──────
        self.optimizer = Adam(model.parameters(), lr=config.INITIAL_LR)
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_DECAY,
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        self._train_meter = ConfusionMeter(
            n_classes=config.NUM_CLASSES,
            ignore_index=config.IGNORE_INDEX,
        )

        # ── Output dirs ───────────────────────────────────────────────────────
        self._ckpt_dir    = Path(config.CHECKPOINTS_DIR) / run_name
        self._results_dir = Path(config.EXPERIMENTS_DIR) / "results" / run_name

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, num_epochs: int) -> None:
        """Full training loop under a single MLflow run."""
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)

        physical_bs = self.train_loader.batch_size or 0
        eff_bs = physical_bs * self.accumulation_steps

        with mlflow.start_run(run_name=self.run_name):

            # ── Git tags ──────────────────────────────────────────────────────
            git_tags = _git_tags()
            if git_tags:
                mlflow.set_tags(git_tags)

            # ── Namespaced params ─────────────────────────────────────────────
            mlflow.log_params({
                # model
                "model/arch":              self.arch,
                "model/encoder":           self.encoder_name,
                "model/encoder_depth":     getattr(self.config, "ENCODER_DEPTH", 5),
                "model/in_channels":       self.config.IN_CHANNELS,
                "model/num_classes":       self.config.NUM_CLASSES,
                # train
                "train/loss":              "weighted_cross_entropy",
                "train/lr":                self.config.INITIAL_LR,
                "train/lr_step":           self.config.LR_STEP_SIZE,
                "train/lr_decay":          self.config.LR_DECAY,
                "train/class_weights":     str(self.config.CLASS_WEIGHTS),
                "train/batch_size":        physical_bs,
                "train/accumulation_steps": self.accumulation_steps,
                "train/batch_size_effective": eff_bs,
                "train/num_epochs":        num_epochs,
                "train/checkpoint_every":  self.checkpoint_every,
                "train/eval_every":        self.eval_every,
                # data
                "data/bands":              str(self.config.MODEL_BANDS),
                "data/num_train_tiles":    len(self.train_loader.dataset),
                "data/num_test_tiles":     len(self.test_loader.dataset),
                # run
                "run/device":              self.device,
            })

            # ── Model stats as metrics at step 0 ─────────────────────────────
            mlflow.log_metrics(_model_stats(self.model), step=0)

            print(f"[MLflow] {self.config.MLFLOW_TRACKING_URI}  "
                  f"experiment: '{self.config.MLFLOW_EXPERIMENT_NAME}'  run: '{self.run_name}'")

            run_start = time.time()

            for epoch in range(1, num_epochs + 1):
                t0 = time.time()
                train_loss, train_metrics = self._train_epoch(epoch)
                train_elapsed = time.time() - t0

                lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()

                # Notify dataset to reshuffle random index for next epoch
                ds = self.train_loader.dataset
                if hasattr(ds, "new_epoch"):
                    ds.new_epoch()

                # ── Log training metrics ──────────────────────────────────────
                n_train = len(self.train_loader.dataset)
                train_log = {
                    "train/loss":                    train_loss,
                    "train/iou/erosion":             train_metrics["iou_erosion"],
                    "train/iou/no_erosion":          train_metrics["iou_no_erosion"],
                    "train/iou/mean":                train_metrics["mean_iou"],
                    "train/f1/erosion":              train_metrics["f1_erosion"],
                    "train/precision/erosion":       train_metrics["precision_erosion"],
                    "train/recall/erosion":          train_metrics["recall_erosion"],
                    "train/time/epoch_duration_min": round(train_elapsed / 60, 3),
                    "train/throughput/tiles_per_sec": round(n_train / train_elapsed, 1),
                    "lr": lr,
                }
                mlflow.log_metrics(train_log, step=epoch)

                print(
                    f"Ep {epoch:4d}/{num_epochs} | "
                    f"loss={train_loss:.4f} | "
                    f"iou_erosion={train_metrics['iou_erosion']:.3f} | "
                    f"f1={train_metrics['f1_erosion']:.3f} | "
                    f"prec={train_metrics['precision_erosion']:.3f} | "
                    f"rec={train_metrics['recall_erosion']:.3f} | "
                    f"lr={lr:.2e} | "
                    f"{train_elapsed:.0f}s"
                )

                # ── Periodic test evaluation ──────────────────────────────────
                if epoch % self.eval_every == 0 or epoch == num_epochs:
                    t1 = time.time()
                    collector = self._eval_epoch(epoch)
                    test_elapsed = time.time() - t1

                    global_m  = collector.compute_global()
                    tile_mean = collector.compute_tile_average()
                    n_test    = len(self.test_loader.dataset)

                    test_log = {
                        "test/loss":                        self._last_test_loss,
                        "test/iou/erosion":                 global_m["iou_erosion"],
                        "test/iou/no_erosion":              global_m["iou_no_erosion"],
                        "test/iou/mean":                    global_m["mean_iou"],
                        "test/f1/erosion":                  global_m["f1_erosion"],
                        "test/precision/erosion":           global_m["precision_erosion"],
                        "test/recall/erosion":              global_m["recall_erosion"],
                        "test/false_alarm/erosion":         global_m["false_alarm_erosion"],
                        "test/miss_rate/erosion":           global_m["miss_rate_erosion"],
                        "test/tile_mean/f1_erosion":        tile_mean.get("tile_mean_f1_erosion", 0),
                        "test/tile_mean/iou_erosion":       tile_mean.get("tile_mean_iou_erosion", 0),
                        "test/time/epoch_duration_min":     round(test_elapsed / 60, 3),
                        "test/throughput/tiles_per_sec":    round(n_test / test_elapsed, 1),
                    }
                    mlflow.log_metrics(test_log, step=epoch)

                    # Save per-tile parquet
                    parquet_path = self._results_dir / f"epoch_{epoch:04d}_test_metrics.parquet"
                    saved = collector.save(parquet_path)
                    mlflow.log_artifact(str(saved), artifact_path="test_metrics")

                    print(
                        f"  TEST  | "
                        f"iou={global_m['iou_erosion']:.3f} | "
                        f"f1={global_m['f1_erosion']:.3f} | "
                        f"prec={global_m['precision_erosion']:.3f} | "
                        f"rec={global_m['recall_erosion']:.3f} | "
                        f"false_alarm={global_m['false_alarm_erosion']:.3f} | "
                        f"{test_elapsed:.0f}s"
                    )

                # ── Checkpoint ────────────────────────────────────────────────
                if epoch % self.checkpoint_every == 0 or epoch == num_epochs:
                    self._save_checkpoint(epoch)

            mlflow.log_metric("run/total_duration_min", round((time.time() - run_start) / 60, 2))

    # ── Private ────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> tuple[float, dict]:
        self.model.train()
        self._train_meter.reset()

        total_loss = 0.0
        n_steps    = 0
        self.optimizer.zero_grad()

        for step, (images, masks, _) in enumerate(
            tqdm(self.train_loader, desc=f"Train {epoch:4d}", leave=False, ncols=100)
        ):
            images = images.to(self.device)
            masks  = masks.to(self.device)

            logits = self.model(images)
            loss   = self.criterion(logits, masks) / self.accumulation_steps
            loss.backward()

            total_loss += loss.item() * self.accumulation_steps
            n_steps    += 1
            self._train_meter.update(logits.detach(), masks)

            if (step + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Flush remaining gradients
        if n_steps % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / max(n_steps, 1), self._train_meter.compute()

    @torch.no_grad()
    def _eval_epoch(self, epoch: int) -> TileMetricsCollector:
        self.model.eval()
        collector = TileMetricsCollector(
            model_name=self.run_name,
            epoch=epoch,
        )

        total_loss = 0.0
        n_steps    = 0

        for images, masks, meta in tqdm(
            self.test_loader, desc="Test  ", leave=False, ncols=100
        ):
            images = images.to(self.device)
            masks  = masks.to(self.device)

            logits = self.model(images)
            loss   = self.criterion(logits, masks)
            total_loss += loss.item()
            n_steps    += 1

            collector.add_batch(logits, masks, meta)

        self._last_test_loss = total_loss / max(n_steps, 1)
        return collector

    def _save_checkpoint(self, epoch: int) -> None:
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self._ckpt_dir / f"epoch_{epoch:04d}.pth"
        torch.save({
            "epoch":      epoch,
            "arch":       self.arch,
            "state_dict": self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "scheduler":  self.scheduler.state_dict(),
        }, path)
        # Checkpoints are kept locally only — Cloud Run rejects large uploads (413)
        mlflow.log_metric("checkpoint/last_saved_epoch", epoch, step=epoch)
        print(f"  → Checkpoint: {path}")
