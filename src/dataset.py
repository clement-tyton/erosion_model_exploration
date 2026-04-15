"""
Dataset for loading erosion tile NPZ files.

Each tile is stored as two NPZ files:
  - imagery_file: shape (H, W, N_bands) — float32 pixel values
  - mask_file:    shape (H, W)          — uint8 labels {0, 1, 14}

Band order in the NPZ matches the 'bands' field in the tile JSON entry.
We select MODEL_BANDS in the correct order, normalize, and remap mask labels.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import (
    DATA_DIR,
    IGNORE_INDEX,
    LABEL_TO_CHANNEL,
    MODEL_BANDS,
    TILES_JSON,
    TRAIN_MEAN,
    TRAIN_STD,
)


def load_tiles_json(path: Path = TILES_JSON) -> list[dict]:
    """Load balanced tiles list from JSON."""
    with open(path) as f:
        data = json.load(f)
    return data["balanced_tiles"]


def filter_tiles(tiles: list[dict], required_bands: list[str] = MODEL_BANDS) -> list[dict]:
    """Keep only tiles that contain all required bands."""
    required = set(required_bands)
    return [t for t in tiles if required.issubset(set(t["bands"]))]


def _remap_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remap raw pixel labels to model channel indices.
      0  → IGNORE_INDEX (background, excluded from metrics)
      1  → 0  (no-erosion)
      14 → 1  (erosion)
    """
    out = np.full_like(mask, fill_value=IGNORE_INDEX, dtype=np.uint8)
    for raw_label, channel_idx in LABEL_TO_CHANNEL.items():
        out[mask == raw_label] = channel_idx
    return out


class TileDataset(Dataset):
    """
    Loads imagery + mask NPZ pairs from DATA_DIR.

    Returns:
        image   : FloatTensor [4, H, W]  — normalised MODEL_BANDS
        mask    : LongTensor  [H, W]     — channel indices (IGNORE_INDEX for background)
        meta    : dict with keys: imagery_file, mask_file, bands, count, tile_idx
    """

    def __init__(
        self,
        tile_entries: list[dict],
        data_dir: Path = DATA_DIR,
        model_bands: list[str] = MODEL_BANDS,
        mean: list[float] = TRAIN_MEAN,
        std: list[float] = TRAIN_STD,
    ):
        self.data_dir = Path(data_dir)
        # Filter to only tiles where both files exist on disk
        present = [
            e for e in tile_entries
            if (self.data_dir / e["imagery_file"]).exists()
            and (self.data_dir / e["mask_file"]).exists()
        ]
        if len(present) < len(tile_entries):
            print(f"[dataset] {len(tile_entries) - len(present)} tiles skipped "
                  f"(files not on disk) — {len(present)} tiles loaded")
        self.entries = present
        self.model_bands = model_bands
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        img_path = self.data_dir / entry["imagery_file"]
        mask_path = self.data_dir / entry["mask_file"]

        # ── Load imagery ──────────────────────────────────────────────────────
        # NPZ stores one key per band (e.g. 'RED', 'GREEN', 'BLUE', 'DSM_NORMALIZED', ...)
        img_npz = np.load(img_path)
        img = np.stack(
            [img_npz[b].astype(np.float32) for b in self.model_bands], axis=-1
        )  # (H, W, 4)

        # Normalise per-band
        img = (img - self.mean) / self.std  # (H, W, 4)

        # To channel-first tensor
        image = torch.from_numpy(img.transpose(2, 0, 1))  # (4, H, W)

        # ── Load mask ─────────────────────────────────────────────────────────
        mask_npz = np.load(mask_path)
        mask_key = list(mask_npz.keys())[0]
        mask_raw = mask_npz[mask_key]

        if mask_raw.ndim == 3:
            # (1, H, W) → (H, W)
            mask_raw = mask_raw.squeeze(0)

        mask = _remap_mask(mask_raw.astype(np.uint8))
        mask_tensor = torch.from_numpy(mask).long()  # (H, W)

        meta = {
            "imagery_file": entry["imagery_file"],
            "mask_file": entry["mask_file"],
            "bands": entry["bands"],
            "count": entry.get("count", 0),
            "tile_idx": idx,
        }

        return image, mask_tensor, meta


def _pad32(x: int) -> int:
    """Round up to nearest multiple of 32 (SMP encoder requirement)."""
    return ((x + 31) // 32) * 32


def collate_pad(batch):
    """
    Custom collate that pads images and masks to the largest tile in the batch,
    rounded up to the nearest multiple of 32 (required by SMP encoders).
    Image padding: 0.0  (normalised black)
    Mask padding:  IGNORE_INDEX (excluded from metrics)
    """
    images, masks, metas = zip(*batch)

    max_h = _pad32(max(img.shape[1] for img in images))
    max_w = _pad32(max(img.shape[2] for img in images))

    padded_images = []
    padded_masks  = []
    for img, mask in zip(images, masks):
        h, w = img.shape[1], img.shape[2]
        pad_h, pad_w = max_h - h, max_w - w
        # F.pad order: (left, right, top, bottom, ...)
        padded_images.append(torch.nn.functional.pad(img,  (0, pad_w, 0, pad_h), value=0.0))
        padded_masks.append( torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), value=IGNORE_INDEX))

    # Merge meta list-of-dicts → dict-of-lists (standard DataLoader behaviour)
    collated_meta = {k: [m[k] for m in metas] for k in metas[0]}

    return torch.stack(padded_images), torch.stack(padded_masks), collated_meta
