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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    StepLR,
)
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
        use_amp: bool = False,
        compile_mode: str | None = None,
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

        # ── Optimizer ─────────────────────────────────────────────────────────
        # SegFormer: AdamW + differential LR (encoder 10× lower than head)
        #            + gradient clipping + warmup+cosine schedule (see _build_scheduler)
        # UNet/others: Adam + StepLR (matches objecttrain)
        if arch.lower() == "segformer":
            head_lr = getattr(config, "SEGFORMER_HEAD_LR",    6e-5)
            enc_lr  = getattr(config, "SEGFORMER_ENCODER_LR", 6e-6)
            encoder_ids    = {id(p) for p in model.encoder.parameters()}
            encoder_params = [p for p in model.parameters() if id(p) in encoder_ids]
            head_params    = [p for p in model.parameters() if id(p) not in encoder_ids]
            self.optimizer = AdamW([
                {"params": encoder_params, "lr": enc_lr},
                {"params": head_params,    "lr": head_lr},
            ], weight_decay=0.01)
            self.grad_clip_norm: float | None = getattr(config, "SEGFORMER_GRAD_CLIP", 0.5)
            self._warmup_ratio: float = getattr(config, "SEGFORMER_WARMUP_RATIO", 0.05)
        else:
            self.optimizer = Adam(model.parameters(), lr=config.INITIAL_LR)
            self.grad_clip_norm = None
            self._warmup_ratio  = 0.0

        # Scheduler is built lazily in train() — SegFormer needs num_epochs for warmup
        self.scheduler = None

        # ── Metrics ───────────────────────────────────────────────────────────
        self._train_meter = ConfusionMeter(
            n_classes=config.NUM_CLASSES,
            ignore_index=config.IGNORE_INDEX,
        )

        # ── Output dirs ───────────────────────────────────────────────────────
        self._ckpt_dir    = Path(config.CHECKPOINTS_DIR) / run_name
        self._results_dir = Path(config.EXPERIMENTS_DIR) / "results" / run_name

        # ── AMP & torch.compile ───────────────────────────────────────────────
        self.use_amp      = use_amp
        self.compile_mode = compile_mode
        self._device_type = "cuda" if "cuda" in str(device) else "cpu"

        # GradScaler: enabled only for CUDA AMP; behaves as no-op otherwise
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(use_amp and self._device_type == "cuda")
        )

        # torch.compile: wraps the forward pass; state_dict/train/eval still work.
        # CUDAGraphs disabled: incompatible with gradient accumulation (buffer overwrite).
        # Triton autotuning (the real gain source) is preserved.
        if compile_mode is not None:
            print(f"[Trainer] torch.compile mode='{compile_mode}' …")
            self.model = torch.compile(self.model, mode=compile_mode)

    # ── Scheduler factory ──────────────────────────────────────────────────────

    def _build_scheduler(self, num_epochs: int):
        """
        SegFormer : linear warmup (5 % of epochs) → cosine annealing to eta_min=1e-8.
        UNet/CNN  : StepLR identical to objecttrain (step_size, gamma from config).
        """
        if self.arch.lower() == "segformer":
            warmup_epochs = max(1, round(self._warmup_ratio * num_epochs))
            cosine_epochs = num_epochs - warmup_epochs
            warmup = LinearLR(
                self.optimizer,
                start_factor=0.01,   # starts at 1 % of target LR
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, cosine_epochs),
                eta_min=1e-8,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            return StepLR(
                self.optimizer,
                step_size=self.config.LR_STEP_SIZE,
                gamma=self.config.LR_DECAY,
            )

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(self, num_epochs: int) -> None:
        """Full training loop under a single MLflow run."""
        mlflow.set_tracking_uri(self.config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config.MLFLOW_EXPERIMENT_NAME)

        physical_bs = self.train_loader.batch_size or 0
        eff_bs = physical_bs * self.accumulation_steps

        self.scheduler = self._build_scheduler(num_epochs)

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
                "train/optimizer":         "adamw" if self.arch.lower() == "segformer" else "adam",
                "train/lr":                self.optimizer.param_groups[-1]["lr"],   # head LR
                "train/lr_backbone":       self.optimizer.param_groups[0]["lr"] if len(self.optimizer.param_groups) > 1 else self.optimizer.param_groups[0]["lr"],
                "train/grad_clip_norm":    self.grad_clip_norm if self.grad_clip_norm is not None else -1,
                "train/scheduler":         "warmup_cosine" if self.arch.lower() == "segformer" else "step_lr",
                "train/warmup_epochs":     max(1, round(self._warmup_ratio * num_epochs)) if self.arch.lower() == "segformer" else 0,
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
                "run/use_amp":             self.use_amp,
                "run/compile_mode":        self.compile_mode or "none",
            })

            # ── Model stats as metrics at step 0 ─────────────────────────────
            mlflow.log_metrics(_model_stats(self.model), step=0)

            print(f"[MLflow] {self.config.MLFLOW_TRACKING_URI}  "
                  f"experiment: '{self.config.MLFLOW_EXPERIMENT_NAME}'  run: '{self.run_name}'")

            run_start = time.time()

            for epoch in range(1, num_epochs + 1):
                t0 = time.time()
                train_loss, train_metrics, grad_stats = self._train_epoch(epoch)
                train_elapsed = time.time() - t0

                lr = self.optimizer.param_groups[-1]["lr"]   # head LR (or only group)
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
                    "train/grad_norm/mean":          round(grad_stats["grad_norm/mean"], 4),
                    "train/grad_norm/max":           round(grad_stats["grad_norm/max"], 4),
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

    def _train_epoch(self, epoch: int) -> tuple[float, dict, dict]:
        self.model.train()
        self._train_meter.reset()

        total_loss = 0.0
        n_steps    = 0
        grad_norms: list[float] = []
        self.optimizer.zero_grad()

        for step, (images, masks, _) in enumerate(
            tqdm(self.train_loader, desc=f"Train {epoch:4d}", leave=False, ncols=100)
        ):
            images = images.to(self.device)
            masks  = masks.to(self.device)

            with torch.autocast(device_type=self._device_type, enabled=self.use_amp):
                logits = self.model(images)
                loss   = self.criterion(logits, masks) / self.accumulation_steps
            self.scaler.scale(loss).backward()

            total_loss += loss.item() * self.accumulation_steps
            n_steps    += 1
            self._train_meter.update(logits.detach(), masks)

            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm if self.grad_clip_norm is not None else float("inf"),
                )
                grad_norms.append(norm.item())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        # Flush remaining gradients
        if n_steps % self.accumulation_steps != 0:
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.grad_clip_norm if self.grad_clip_norm is not None else float("inf"),
            )
            grad_norms.append(norm.item())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        grad_stats = {
            "grad_norm/mean": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
            "grad_norm/max":  max(grad_norms) if grad_norms else 0.0,
        }
        return total_loss / max(n_steps, 1), self._train_meter.compute(), grad_stats

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

            with torch.autocast(device_type=self._device_type, enabled=self.use_amp):
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
