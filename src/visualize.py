"""
Generate 2×2 PNG visualisations for tiles ranked by metric.

Usage:
    python -m src.visualize [--n 200] [--sort f1_erosion_asc]

Flags:
    --n      Number of tiles to visualise (default 200)
    --sort   Sorting key. Options:
               f1_erosion_asc    (default) — worst erosion F1 first
               f1_erosion_desc              — best erosion F1 first
               f1_no_erosion_asc
               f1_no_erosion_desc
               iou_erosion_asc
               n_erosion_pixels_desc        — tiles with most erosion first

Each PNG is a 2×2 grid:
    [RGB image]          [DSM_NORMALIZED]
    [Predicted mask]     [True mask]

Saved to output/tiles/<imagery_file_stem>.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    IGNORE_INDEX,
    METRICS_CSV,
    MODEL_BANDS,
    MODEL_PATH,
    OUTPUT_DIR,
    TILES_DIR,
    TILES_JSON,
    TRAIN_MEAN,
    TRAIN_STD,
)
from src.dataset import TileDataset, filter_tiles, load_tiles_json
from src.model import load_model

import torch


# ── Colour map for masks ──────────────────────────────────────────────────────
# channel 0 = no-erosion → green
# channel 1 = erosion    → red
# IGNORE_INDEX            → light grey
MASK_COLORS = {
    0:            (0.2, 0.7, 0.2),   # no-erosion: green
    1:            (0.9, 0.1, 0.1),   # erosion: red
    IGNORE_INDEX: (0.8, 0.8, 0.8),   # background: grey
}

def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) channel-index mask → (H, W, 3) RGB float image."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for val, color in MASK_COLORS.items():
        m = mask == val
        rgb[m] = color
    return rgb


def _denorm_rgb(img_chw: np.ndarray) -> np.ndarray:
    """
    Reverse normalisation and return (H, W, 3) uint8 RGB.
    img_chw: (4, H, W) normalised float — bands in MODEL_BANDS order.
    We take channels 0,1,2 = RED, GREEN, BLUE.
    """
    mean = np.array(TRAIN_MEAN[:3], dtype=np.float32)
    std  = np.array(TRAIN_STD[:3],  dtype=np.float32)
    rgb = img_chw[:3].transpose(1, 2, 0) * std + mean  # (H, W, 3)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _denorm_dsm(img_chw: np.ndarray) -> np.ndarray:
    """Return (H, W) float DSM_NORMALIZED channel, denormalised."""
    dsm_idx = MODEL_BANDS.index("DSM_NORMALIZED")
    mean = TRAIN_MEAN[dsm_idx]
    std  = TRAIN_STD[dsm_idx]
    dsm = img_chw[dsm_idx] * std + mean   # (H, W)
    return dsm


def save_tile_png(
    imagery_file: str,
    img_chw: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    metrics: dict,
    out_path: Path,
):
    """
    Save a 2×2 matplotlib figure for one tile.

    Args:
        imagery_file: filename string (used for title)
        img_chw:      (4, H, W) normalised float tensor converted to numpy
        pred_mask:    (H, W) predicted channel indices
        true_mask:    (H, W) true channel indices (may contain IGNORE_INDEX)
        metrics:      dict with at least f1_erosion, n_erosion_pixels
        out_path:     destination PNG path
    """
    rgb = _denorm_rgb(img_chw)
    dsm = _denorm_dsm(img_chw)
    pred_rgb = _mask_to_rgb(pred_mask)
    true_rgb = _mask_to_rgb(true_mask)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        f"{imagery_file}\n"
        f"F1 erosion: {metrics.get('f1_erosion', 0):.3f}  "
        f"IOU erosion: {metrics.get('iou_erosion', 0):.3f}  "
        f"N erosion px: {metrics.get('n_erosion_pixels', 0):,}",
        fontsize=9,
    )

    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(dsm, cmap="terrain")
    axes[0, 1].set_title("DSM_NORMALIZED")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(pred_rgb)
    axes[1, 0].set_title("Predicted mask")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(true_rgb)
    axes[1, 1].set_title("True mask")
    axes[1, 1].axis("off")

    # Legend
    legend_patches = [
        mpatches.Patch(color=MASK_COLORS[0], label="No erosion"),
        mpatches.Patch(color=MASK_COLORS[1], label="Erosion"),
        mpatches.Patch(color=MASK_COLORS[IGNORE_INDEX], label="Background"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _overlay(ax, base_img, mask, color, alpha: float = 0.20, contour: bool = True):
    """
    Draw base_img then draw only the contour of erosion pixels (mask==1).
    No fill — base image shows through completely.
    """
    ax.imshow(base_img, cmap="terrain" if base_img.ndim == 2 else None)

    ero = mask == 1
    if ero.any():
        ax.contour(ero.astype(np.uint8), levels=[0.5], colors=[color], linewidths=[1.5])


def save_tile_overlay_png(
    imagery_file: str,
    img_chw: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    metrics: dict,
    out_path: Path,
):
    """
    2×2 overlay visualisation:
        [RGB + pred overlay]    [RGB + true overlay]
        [DSM + pred overlay]    [DSM + true overlay]

    Erosion  → red  (filled transparent + contour)
    No-erosion → blue (faint fill)
    """
    rgb = _denorm_rgb(img_chw)
    dsm = _denorm_dsm(img_chw)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{imagery_file}\n"
        f"F1 erosion: {metrics.get('f1_erosion', 0):.3f}  "
        f"IOU: {metrics.get('iou_erosion', 0):.3f}  "
        f"Recall: {metrics.get('recall_erosion', 0):.3f}  "
        f"Precision: {metrics.get('precision_erosion', 0):.3f}  "
        f"N erosion px: {metrics.get('n_erosion_pixels', 0):,}",
        fontsize=9,
    )

    BLUE = (0.1, 0.4, 1.0)   # prediction
    RED  = (1.0, 0.1, 0.1)   # ground truth

    _overlay(axes[0, 0], rgb, pred_mask, color=BLUE)
    axes[0, 0].set_title("RGB — Prediction (blue)", fontsize=9)
    axes[0, 0].axis("off")

    _overlay(axes[0, 1], rgb, true_mask, color=RED)
    axes[0, 1].set_title("RGB — Ground truth (red)", fontsize=9)
    axes[0, 1].axis("off")

    _overlay(axes[1, 0], dsm, pred_mask, color=BLUE)
    axes[1, 0].set_title("DSM — Prediction (blue)", fontsize=9)
    axes[1, 0].axis("off")

    _overlay(axes[1, 1], dsm, true_mask, color=RED)
    axes[1, 1].set_title("DSM — Ground truth (red)", fontsize=9)
    axes[1, 1].axis("off")

    legend_patches = [
        mpatches.Patch(color=BLUE, label="Prediction — erosion"),
        mpatches.Patch(color=RED,  label="Ground truth — erosion"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

SORT_OPTIONS = {
    "f1_erosion_asc":         ("f1_erosion",        True),
    "f1_erosion_desc":        ("f1_erosion",        False),
    "f1_no_erosion_asc":      ("f1_no_erosion",     True),
    "f1_no_erosion_desc":     ("f1_no_erosion",     False),
    "iou_erosion_asc":        ("iou_erosion",       True),
    "n_erosion_pixels_desc":  ("n_erosion_pixels",  False),
}


def run_visualize(n: int = 200, sort: str = "f1_erosion_asc"):
    if not METRICS_CSV.exists():
        print(
            f"[visualize] ERROR: {METRICS_CSV} not found.\n"
            "Run inference first:  python -m src.evaluate"
        )
        return

    sort_col, ascending = SORT_OPTIONS.get(sort, ("f1_erosion", True))
    df = pd.read_csv(METRICS_CSV).sort_values(sort_col, ascending=ascending).head(n)
    print(f"[visualize] Generating {len(df)} PNGs sorted by {sort_col} (asc={ascending})")

    # Build lookup: imagery_file → metrics dict
    metrics_by_file = df.set_index("imagery_file").to_dict("index")

    # Build dataset from the selected subset
    all_tiles = load_tiles_json()
    all_tiles_map = {t["imagery_file"]: t for t in all_tiles}
    selected_tiles = [all_tiles_map[f] for f in df["imagery_file"] if f in all_tiles_map]

    dataset = TileDataset(selected_tiles, data_dir=DATA_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[visualize] Loading model on {device}")
    model = load_model(MODEL_PATH, device=device)

    TILES_DIR.mkdir(parents=True, exist_ok=True)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=2
    )

    with torch.no_grad():
        for images, masks, metas in tqdm(loader, desc="Generating PNGs"):
            images_dev = images.to(device)
            probs = model(images_dev)
            preds = probs.argmax(dim=1).cpu().numpy()  # (B, H, W)
            masks_np = masks.numpy()                   # (B, H, W)
            images_np = images.numpy()                 # (B, 4, H, W)

            for i in range(images.shape[0]):
                fname = metas["imagery_file"][i]
                stem = Path(fname).stem
                out_path = TILES_DIR / f"{stem}.png"

                if out_path.exists():
                    continue  # skip already generated

                tile_metrics = metrics_by_file.get(fname, {})
                save_tile_png(
                    imagery_file=fname,
                    img_chw=images_np[i],
                    pred_mask=preds[i],
                    true_mask=masks_np[i],
                    metrics=tile_metrics,
                    out_path=out_path,
                )

    print(f"[visualize] Done — PNGs saved to {TILES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Generate tile visualisation PNGs")
    parser.add_argument("--n",    type=int, default=200,
                        help="Number of tiles to visualise")
    parser.add_argument("--sort", type=str, default="f1_erosion_asc",
                        choices=list(SORT_OPTIONS.keys()),
                        help="Sort order for selecting tiles")
    args = parser.parse_args()
    run_visualize(n=args.n, sort=args.sort)


if __name__ == "__main__":
    main()
