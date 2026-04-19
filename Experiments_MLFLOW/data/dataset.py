"""
PyTorch datasets for Experiments_MLFLOW — aligned with production objecttrain code.

Augmentation pipeline (matches objecttrain exactly):
  Train:
    Normalize → PadIfNeeded(384,384) → RandomCrop(256,256)
    → OneOf([HFlip, VFlip], p=0.66)
    → OneOf([ElasticTransform, GridDistortion, ShiftScaleRotate(rot180,scale±20%)], p=0.6)

  Val/Test:
    Normalize → PadIfNeeded(384,384) → RandomCrop(384,384)

preprocess_before_augmentation=True (from config): normalise first, then spatial augs.
optional_band_drop: NOT active (optional_bands absent from production config → []).

Mask format:
  Raw mask labels: 0=background, 1=no-erosion, 14=erosion
  Remapped:        0→IGNORE(255), 1→0, 14→1  (LongTensor for CrossEntropyLoss)

TrainDataset:
  Source: metadata/balanced_tiles.json — {"balanced_tiles": [{…, count: N}]}
  Tile duplication already encoded via `count` in the JSON.
  Optional samples_per_train_epoch: limits tiles per epoch (see objecttrain Dataset).
  Call new_epoch() each epoch to refresh the random index when using samples_per_train_epoch.

TestDataset:
  Source: EROSION_DATASET_TEST_METADATA.json — flat list.
  No duplication, val augmentation only.
"""

import json
import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import Dataset

# ── Default paths ──────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = Path(__file__).parent.parent
METADATA_DIR    = EXPERIMENTS_DIR / "metadata"

DEFAULT_TRAIN_DATA_DIR = EXPERIMENTS_DIR / "data" / "train_data"
DEFAULT_TEST_DATA_DIR  = EXPERIMENTS_DIR / "data" / "test_data"

BALANCED_TILES_JSON = METADATA_DIR / "balanced_tiles.json"
TEST_METADATA_JSON  = METADATA_DIR / "EROSION_DATASET_TEST_METADATA.json"

MODEL_BANDS  = ["RED", "GREEN", "BLUE", "DSM_NORMALIZED"]
TRAIN_MEAN   = [150.73301134918557, 123.75755228360018, 92.57823716578613, -9.734063808604613]
TRAIN_STD    = [39.721974708734216, 34.06117915518031, 30.092062243775406, 4.684211737168346]
IGNORE_INDEX = 255
LABEL_TO_CHANNEL = {0: IGNORE_INDEX, 1: 0, 14: 1}


# ── Augmentation factories ─────────────────────────────────────────────────────

def _get_training_augmentation(
    mean: list[float],
    std: list[float],
    crop_size: int = 256,
) -> A.Compose:
    """
    Production training augmentation (objecttrain._get_training_augmentation).
    preprocess_before_augmentation=True → Normalize first, spatial augs after.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # ShiftScaleRotate deprecation
        return A.Compose([
            # 1. Normalise (before spatial augs — matches preprocess_before_augmentation=True)
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            # 2. Ensure minimum size for crop
            A.PadIfNeeded(
                min_height=384, min_width=384,
                border_mode=0, fill=0,
                p=1.0,
            ),
            # 3. Random crop to training size
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
            # 4. Flip (66% chance, H or V)
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
            ], p=0.66),
            # 5. Geometric distortion (60% chance)
            A.OneOf([
                A.ElasticTransform(border_mode=0, fill=0, p=1.0),
                A.GridDistortion(distort_limit=0.4, border_mode=0, fill=0, p=1.0),
                A.ShiftScaleRotate(
                    shift_limit=0.0,
                    scale_limit=(-0.2, 0.2),
                    rotate_limit=180,
                    border_mode=0,
                    fill=0,
                    p=1.0,
                ),
            ], p=0.6),
        ])


def _get_validation_augmentation(
    mean: list[float],
    std: list[float],
    pad_size: int = 384,
) -> A.Compose:
    """
    Production validation augmentation (objecttrain._get_validation_augmentation).
    Pad to pad_size, then crop to pad_size → ensures shape divisible by 32.
    """
    return A.Compose([
        A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
        A.PadIfNeeded(min_height=pad_size, min_width=pad_size, border_mode=0, fill=0, p=1.0),
        A.RandomCrop(height=pad_size, width=pad_size, p=1.0),
    ])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remap raw mask pixel labels → channel indices.
      0  → IGNORE_INDEX (255) — background, excluded from loss/metrics
      1  → 0  (no-erosion)
      14 → 1  (erosion)
    """
    out = np.full_like(mask, fill_value=IGNORE_INDEX, dtype=np.uint8)
    for raw_label, channel_idx in LABEL_TO_CHANNEL.items():
        out[mask == raw_label] = channel_idx
    return out


def _load_image(npz_path: Path, model_bands: list[str]) -> np.ndarray:
    """Load imagery NPZ, return selected bands as float32 (H, W, C)."""
    npz = np.load(npz_path)
    return np.stack([npz[b].astype(np.float32) for b in model_bands], axis=-1)


def _load_mask(npz_path: Path) -> np.ndarray:
    """Load mask NPZ, return (H, W) uint8."""
    npz = np.load(npz_path)
    mask = npz[list(npz.keys())[0]]
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask.astype(np.uint8)


def collate_pad(batch):
    """
    Collate fn: pad to largest tile in batch, rounded to multiple of 32.
    With fixed-size augmentations this is usually a no-op but kept for safety.
    """
    images, masks, metas = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # Round up to multiple of 32
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32

    padded_images, padded_masks = [], []
    for img, mask in zip(images, masks):
        ph = max_h - img.shape[1]
        pw = max_w - img.shape[2]
        padded_images.append(
            torch.nn.functional.pad(img,  (0, pw, 0, ph), value=0.0)
        )
        padded_masks.append(
            torch.nn.functional.pad(mask, (0, pw, 0, ph), value=IGNORE_INDEX)
        )

    collated_meta = {k: [m[k] for m in metas] for k in metas[0]}
    return torch.stack(padded_images), torch.stack(padded_masks), collated_meta


# ── TrainDataset ──────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """
    Loads balanced training tiles from metadata/balanced_tiles.json.

    Tile duplication is pre-encoded in the JSON via the `count` field —
    each tile is repeated `count` times in the entries list.

    samples_per_train_epoch:
        If set, limits the number of tiles per epoch (matches objecttrain behaviour).
        Call new_epoch() after each epoch to refresh the random index.
    """

    def __init__(
        self,
        balanced_tiles_json: Path = BALANCED_TILES_JSON,
        data_dir: Path = DEFAULT_TRAIN_DATA_DIR,
        model_bands: list[str] = MODEL_BANDS,
        mean: list[float] = TRAIN_MEAN,
        std: list[float] = TRAIN_STD,
        train: bool = True,
        samples_per_train_epoch: int | None = None,
    ):
        self.data_dir    = Path(data_dir)
        self.model_bands = model_bands
        self.train       = train
        self._samples_per_train_epoch = samples_per_train_epoch
        self._rng = np.random.default_rng()

        # ── Build augmentation transform ──────────────────────────────────────
        if train:
            self.transform = _get_training_augmentation(mean, std)
        else:
            self.transform = _get_validation_augmentation(mean, std)

        # ── Load tile list (expand by count) ──────────────────────────────────
        with open(balanced_tiles_json) as f:
            raw_tiles = json.load(f)["balanced_tiles"]

        entries: list[dict] = []
        skipped = 0
        for tile in raw_tiles:
            if not (self.data_dir / tile["imagery_file"]).exists() or \
               not (self.data_dir / tile["mask_file"]).exists():
                skipped += 1
                continue
            count = int(tile.get("count", 1))
            for _ in range(count):
                entries.append(tile)

        if skipped:
            print(f"[TrainDataset] {skipped} source tiles skipped (not on disk) — "
                  f"{len(entries):,} entries total (with duplication)")
        self._all_entries = entries

        # ── Random index for samples_per_train_epoch ──────────────────────────
        self._random_idx = (
            self._generate_random_idx() if self._samples_per_train_epoch else None
        )

    # ── Epoch management (call between epochs) ────────────────────────────────

    def new_epoch(self) -> None:
        """Regenerate random tile index for next epoch (when samples_per_train_epoch is set)."""
        if self._random_idx is not None:
            self._random_idx = self._generate_random_idx()

    def _generate_random_idx(self) -> np.ndarray:
        n = len(self._all_entries)
        size = self._samples_per_train_epoch
        return self._rng.choice(n, size=size, replace=(n < size))

    # ── Dataset interface ─────────────────────────────────────────────────────

    @property
    def _entries(self) -> list[dict]:
        return self._all_entries

    def __len__(self) -> int:
        if self._samples_per_train_epoch and self.train:
            return self._samples_per_train_epoch
        return len(self._all_entries)

    def __getitem__(self, i: int):
        # Remap index through random sample if samples_per_train_epoch active
        if self._random_idx is not None:
            i = int(self._random_idx[i])

        entry = self._all_entries[i]

        img  = _load_image(self.data_dir / entry["imagery_file"], self.model_bands)  # (H,W,C) float32
        mask = _load_mask(self.data_dir / entry["mask_file"])                         # (H,W) uint8

        # Remap raw labels before augmentation so spatial transforms stay consistent
        mask = _remap_mask(mask)   # {0→255, 1→0, 14→1}

        # Apply augmentation (normalise + spatial)
        result  = self.transform(image=img, mask=mask)
        img_out = result["image"]   # (H,W,C) float32, normalised
        msk_out = result["mask"]    # (H,W) uint8, remapped labels preserved

        image_t = torch.from_numpy(img_out.transpose(2, 0, 1))   # (C,H,W)
        mask_t  = torch.from_numpy(msk_out.astype(np.int64))     # (H,W) long

        meta = {
            "imagery_file": entry["imagery_file"],
            "mask_file":    entry["mask_file"],
            "bands":        entry.get("bands", self.model_bands),
            "count":        entry.get("count", 1),
            "tile_idx":     i,
        }
        return image_t, mask_t, meta


# ── TestDataset ───────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    """
    Loads test tiles from EROSION_DATASET_TEST_METADATA.json (flat list).
    No tile duplication. Validation augmentation only.
    Field: `imagery_bands` (not `bands` like train).
    """

    def __init__(
        self,
        test_metadata_json: Path = TEST_METADATA_JSON,
        data_dir: Path = DEFAULT_TEST_DATA_DIR,
        model_bands: list[str] = MODEL_BANDS,
        mean: list[float] = TRAIN_MEAN,
        std: list[float] = TRAIN_STD,
    ):
        self.data_dir    = Path(data_dir)
        self.model_bands = model_bands
        self.transform   = _get_validation_augmentation(mean, std)

        with open(test_metadata_json) as f:
            raw_entries = json.load(f)

        entries: list[dict] = []
        skipped = 0
        for entry in raw_entries:
            if not (self.data_dir / entry["imagery_file"]).exists() or \
               not (self.data_dir / entry["mask_file"]).exists():
                skipped += 1
                continue
            entries.append(entry)

        if skipped:
            print(f"[TestDataset] {skipped} entries skipped (not on disk) — "
                  f"{len(entries):,} loaded")
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        entry = self.entries[i]

        img  = _load_image(self.data_dir / entry["imagery_file"], self.model_bands)
        mask = _load_mask(self.data_dir / entry["mask_file"])
        mask = _remap_mask(mask)

        result  = self.transform(image=img, mask=mask)
        img_out = result["image"]
        msk_out = result["mask"]

        image_t = torch.from_numpy(img_out.transpose(2, 0, 1))
        mask_t  = torch.from_numpy(msk_out.astype(np.int64))

        meta = {
            "imagery_file": entry["imagery_file"],
            "mask_file":    entry["mask_file"],
            "bands":        entry.get("imagery_bands", self.model_bands),
            "capture_id":   entry.get("capture_id", ""),
            "capture_name": entry.get("capture_name", ""),
            "tile_idx":     i,
        }
        return image_t, mask_t, meta
