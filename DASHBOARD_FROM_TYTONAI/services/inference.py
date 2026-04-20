"""
PNG generation helpers for train and test tiles.

All session-specific state (model, device, tile_map_dict, etc.) is passed
explicitly so these functions can be called from any module.

Exports
-------
_MODEL_BANDS, _TRAIN_MEAN, _TRAIN_STD, _IGNORE_IDX
png_path, generate_pngs, load_test_tile, generate_test_pngs
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
_MODEL_BANDS = ["RED", "GREEN", "BLUE", "DSM_NORMALIZED"]
_TRAIN_MEAN  = np.array([150.73301134918557, 123.75755228360018,
                          92.57823716578613,  -9.734063808604613], dtype=np.float32)
_TRAIN_STD   = np.array([39.721974708734216,  34.06117915518031,
                          30.092062243775406,   4.684211737168346], dtype=np.float32)
_IGNORE_IDX  = 255


# ── PNG path helper ───────────────────────────────────────────────────────────
def png_path(imagery_file: str, style: str, model_stem: str, subdir: str = "") -> Path:
    from src.config import TILES_DIR
    stem   = Path(imagery_file).stem
    suffix = "_overlay" if style == "overlay" else ""
    folder = TILES_DIR / (subdir if subdir else model_stem)
    return folder / f"{stem}{suffix}.png"


# ── Train tile PNG generation ─────────────────────────────────────────────────
def generate_pngs(
    imagery_file: str,
    metrics_row: dict,
    *,
    model,
    device,
    tile_map_dict: dict,
    data_dir: Path,
    model_stem: str,
) -> tuple[Path, Path]:
    """Generate side-by-side and overlay PNGs for a training tile."""
    from src.dataset import TileDataset
    from src.visualize import save_tile_overlay_png, save_tile_png

    p1 = png_path(imagery_file, "masks",   model_stem)
    p2 = png_path(imagery_file, "overlay", model_stem)

    if p1.exists() and p2.exists():
        return p1, p2

    entry = tile_map_dict.get(imagery_file)
    if entry is None:
        return p1, p2

    ds = TileDataset([entry], data_dir=data_dir)
    image, mask, _ = ds[0]

    _, h, w = image.shape
    ph = ((h + 31) // 32) * 32
    pw = ((w + 31) // 32) * 32
    img_padded = torch.nn.functional.pad(image, (0, pw - w, 0, ph - h))

    with torch.no_grad():
        prob = model(img_padded.unsqueeze(0).to(device))
    pred = prob.argmax(dim=1).squeeze(0).cpu().numpy()[:h, :w]

    img_np  = image.numpy()
    mask_np = mask.numpy()
    p1.parent.mkdir(parents=True, exist_ok=True)

    if not p1.exists():
        save_tile_png(imagery_file, img_np, pred, mask_np, metrics_row, p1)
    if not p2.exists():
        save_tile_overlay_png(imagery_file, img_np, pred, mask_np, metrics_row, p2)
    return p1, p2


# ── Test tile loading ─────────────────────────────────────────────────────────
def load_test_tile(imagery_file: str, mask_file: str, test_data_dir: Path):
    """
    Load a test tile from test_data_dir.
    Returns (img_chw float32 [4,H,W] normalised, mask_hw uint8 remapped).
    """
    img_npz  = np.load(test_data_dir / imagery_file)
    img_hw_c = np.stack([img_npz[b].astype(np.float32) for b in _MODEL_BANDS], axis=-1)
    img_norm = (img_hw_c - _TRAIN_MEAN) / _TRAIN_STD
    img_chw  = img_norm.transpose(2, 0, 1)  # (4, H, W)

    mask_npz = np.load(test_data_dir / mask_file)
    mask_raw = mask_npz[list(mask_npz.keys())[0]]
    if mask_raw.ndim == 3:
        mask_raw = mask_raw.squeeze(0)
    mask_raw = mask_raw.astype(np.uint8)

    mask = np.full_like(mask_raw, _IGNORE_IDX)
    mask[mask_raw == 1]  = 0
    mask[mask_raw == 14] = 1
    return img_chw, mask


# ── Test tile PNG generation ──────────────────────────────────────────────────
def generate_test_pngs(
    imagery_file: str,
    mask_file: str,
    metrics_row: dict,
    *,
    model,
    device,
    model_stem: str,
    test_data_dir: Path,
) -> tuple[Path, Path]:
    """Generate PNGs for a test tile. Cached in output/tiles/test_<model_stem>/."""
    from src.visualize import save_tile_overlay_png, save_tile_png

    subdir = f"test_{model_stem}"
    p1 = png_path(imagery_file, "masks",   model_stem, subdir)
    p2 = png_path(imagery_file, "overlay", model_stem, subdir)

    if p1.exists() and p2.exists():
        return p1, p2

    if not (test_data_dir / imagery_file).exists():
        return p1, p2

    img_chw, mask = load_test_tile(imagery_file, mask_file, test_data_dir)

    img_t = torch.from_numpy(img_chw)
    _, h, w = img_t.shape
    ph = ((h + 31) // 32) * 32
    pw = ((w + 31) // 32) * 32
    img_padded = torch.nn.functional.pad(img_t, (0, pw - w, 0, ph - h))

    with torch.no_grad():
        prob = model(img_padded.unsqueeze(0).to(device))
    pred = prob.argmax(dim=1).squeeze(0).cpu().numpy()[:h, :w]

    p1.parent.mkdir(parents=True, exist_ok=True)
    if not p1.exists():
        save_tile_png(imagery_file, img_chw, pred, mask, metrics_row, p1)
    if not p2.exists():
        save_tile_overlay_png(imagery_file, img_chw, pred, mask, metrics_row, p2)
    return p1, p2
