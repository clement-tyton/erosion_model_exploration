# %%
# ── 0. Imports & working directory ───────────────────────────────────────────
# Make sure the repo root is on sys.path so `src.*` imports work.
import sys
from pathlib import Path

# ROOT = Path(__file__).parent.parent

ROOT = Path("./")
print("ROOT:", ROOT)
print("Python:", sys.version)

# %%
# ── 1. Config ─────────────────────────────────────────────────────────────────
from src.config import (
    TILES_JSON, DATA_DIR, MODEL_PATH, OUTPUT_DIR,
    MODEL_BANDS, TRAIN_MEAN, TRAIN_STD,
    CLASS_LIST, LABEL_TO_CHANNEL, IGNORE_INDEX,
    ENCODER, BATCH_SIZE,
)

print("TILES_JSON  :", TILES_JSON, "| exists:", TILES_JSON.exists())
print("DATA_DIR    :", DATA_DIR,   "| exists:", DATA_DIR.exists())
print("MODEL_PATH  :", MODEL_PATH, "| exists:", MODEL_PATH.exists())
print("MODEL_BANDS :", MODEL_BANDS)
print("TRAIN_MEAN  :", TRAIN_MEAN)
print("TRAIN_STD   :", TRAIN_STD)
print("CLASS_LIST  :", CLASS_LIST)
print("LABEL_TO_CHANNEL:", LABEL_TO_CHANNEL)
print("IGNORE_INDEX:", IGNORE_INDEX)

# %%
# ── 2. Dataset — load JSON & inspect ─────────────────────────────────────────
from src.dataset import load_tiles_json, filter_tiles

all_tiles = load_tiles_json()
print(f"Total tiles in JSON: {len(all_tiles)}")
print("First entry:", all_tiles[0])

tiles = filter_tiles(all_tiles)
print(f"\nTiles with all MODEL_BANDS: {len(tiles)}")
print("First filtered entry:", tiles[0])

# %%
# ── 3. Dataset — load a single NPZ tile ──────────────────────────────────────
# This will fail if data/training_data/ is not yet populated.
from src.dataset import TileDataset
import numpy as np

dataset = TileDataset(tiles, data_dir=DATA_DIR)
print(f"Dataset length: {len(dataset)}")

# Try the first tile
try:
    image, mask, meta = dataset[0]
    print("image shape :", image.shape)   # expect (4, H, W)
    print("image dtype :", image.dtype)
    print("mask shape  :", mask.shape)    # expect (H, W)
    print("mask unique values:", mask.unique().tolist())
    print("meta:", meta)
except FileNotFoundError as e:
    print(f"[SKIP] NPZ not found — download tiles first.\n  {e}")


import numpy as np
from pathlib import Path

path = Path("data/training_data") / "image_238ea38e-6e55-46be-8285-4592f1ba47aa_1536_1536.npz"
npz = np.load(path)
for k in npz.keys():
    print(f"key={k!r}  shape={npz[k].shape}  dtype={npz[k].dtype}")



# %%
# ── 4. Dataset — check normalisation ─────────────────────────────────────────
# Verify mean ≈ 0, std ≈ 1 per channel after normalisation.
try:
    image, mask, meta = dataset[0]
    img_np = image.numpy()  # (4, H, W)
    for i, band in enumerate(MODEL_BANDS):
        ch = img_np[i]
        print(f"  {band:20s}  mean={ch.mean():+.3f}  std={ch.std():.3f}")
except Exception as e:
    print(f"[SKIP] {e}")

# %%
# ── 5. Model — load checkpoint ────────────────────────────────────────────────
import torch
from src.model import load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

try:
    model = load_model(MODEL_PATH, device=device)
    print("Model loaded OK")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
except FileNotFoundError as e:
    print(f"[SKIP] Model file missing.\n  {e}")
    model = None

# %%
# ── 6. Model — single forward pass ───────────────────────────────────────────
# Run one tile through the model and inspect output shape / value range.
try:
    assert model is not None, "Model not loaded"
    image, mask, meta = dataset[0]
    # image.size()
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))  # (1, 2, H, W)

    print("Output shape :", out.shape)
    print("Output min   :", out.min().item())
    print("Output max   :", out.max().item())
    print("Sum per pixel (should be ~1.0 with softmax):", out.sum(dim=1).mean().item())

    pred = out.argmax(dim=1).squeeze(0).cpu()  # (H, W)
    print("Prediction unique:", pred.unique().tolist())
    n_erosion_pred = (pred == 1).sum().item()
    n_total        = pred.numel()
    print(f"Predicted erosion: {n_erosion_pred:,} / {n_total:,} px ({100*n_erosion_pred/n_total:.1f}%)")

except Exception as e:
    print(f"[SKIP] {e}")

# %%
# ── 7. Metrics — single tile ──────────────────────────────────────────────────
from src.evaluate import compute_tile_metrics
import numpy as np

try:
    image, mask, meta = dataset[0]

    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy()

    metrics = compute_tile_metrics(pred, mask.numpy())
    for k, v in metrics.items():
        print(f"  {k:30s}: {v}")

except Exception as e:
    print(f"[SKIP] {e}")

# %%
# ── 8. Metrics — batch of 8 tiles ────────────────────────────────────────────
from torch.utils.data import DataLoader
from src.dataset import collate_pad

try:
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate_pad)
    images_b, masks_b, metas_b = next(iter(loader))

    with torch.no_grad():
        out_b = model(images_b.to(device))
    preds_b = out_b.argmax(dim=1).cpu().numpy()

    print(f"Batch images shape: {images_b.shape}")
    print(f"Batch masks  shape: {masks_b.shape}")
    print(f"Batch preds  shape: {preds_b.shape}")

    for i in range(len(images_b)):
        m = compute_tile_metrics(preds_b[i], masks_b[i].numpy())
        print(f"  tile {i}: f1_erosion={m['f1_erosion']:.3f}  f1_no_erosion={m['f1_no_erosion']:.3f}"
              f"  n_erosion_px={m['n_erosion_pixels']:,}")

except Exception as e:
    print(f"[SKIP] {e}")

# %%
# ── 9. Visualise — single tile PNG ───────────────────────────────────────────
from src.visualize import save_tile_png, save_tile_overlay_png
from src.config import TILES_DIR
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    image, mask, meta = dataset[0]

    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy()

    metrics = compute_tile_metrics(pred, mask.numpy())
    TILES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Style 1: side-by-side masks ──────────────────────────────────────────
    out_path = TILES_DIR / "test_tile.png"
    save_tile_png(imagery_file=meta["imagery_file"], img_chw=image.numpy(),
                  pred_mask=pred, true_mask=mask.numpy(),
                  metrics=metrics, out_path=out_path)
    print(f"Style 1 saved: {out_path}")
    plt.figure(figsize=(12, 12))
    plt.imshow(mpimg.imread(str(out_path)))
    plt.axis("off"); plt.tight_layout(); plt.show()

    # ── Style 2: overlay on RGB/DSM ──────────────────────────────────────────
    out_path2 = TILES_DIR / "test_tile_overlay.png"
    save_tile_overlay_png(imagery_file=meta["imagery_file"], img_chw=image.numpy(),
                          pred_mask=pred, true_mask=mask.numpy(),
                          metrics=metrics, out_path=out_path2)
    print(f"Style 2 saved: {out_path2}")
    plt.figure(figsize=(14, 12))
    plt.imshow(mpimg.imread(str(out_path2)))
    plt.axis("off"); plt.tight_layout(); plt.show()

except Exception as e:
    print(f"[SKIP] {e}")

# %%
# ── 10. NPZ introspection — peek inside a raw file ──────────────────────────
# Useful if shapes / keys look unexpected in steps 3-4.
try:
    first_tile = tiles[0]
    img_path  = DATA_DIR / first_tile["imagery_file"]
    mask_path = DATA_DIR / first_tile["mask_file"]

    img_npz  = np.load(img_path)
    mask_npz = np.load(mask_path)

    print("=== Imagery NPZ ===")
    for k in img_npz.keys():
        arr = img_npz[k]
        if np.issubdtype(arr.dtype, np.number):
            stats = f"min={arr.min():.2f}  max={arr.max():.2f}"
        else:
            stats = f"values={arr.tolist()}"
        print(f"  key={k!r}  shape={arr.shape}  dtype={arr.dtype}  {stats}")

    print("\n=== Mask NPZ ===")
    for k in mask_npz.keys():
        arr = mask_npz[k]
        unique, counts = np.unique(arr, return_counts=True)
        print(f"  key={k!r}  shape={arr.shape}  dtype={arr.dtype}  "
              f"unique values: {dict(zip(unique.tolist(), counts.tolist()))}")

    print("\nTile bands field:", first_tile["bands"])

except FileNotFoundError as e:
    print(f"[SKIP] NPZ files not downloaded yet.\n  {e}")

# ── Geo info — bounding box + EPSG from NPZ tiles ────────────────────────────
import numpy as np
from pathlib import Path

def inspect_tile_geo(npz_path: Path):
    npz = np.load(npz_path)

    srid = int(np.asarray(npz["SRID"]).flat[0])      # scalar or 1-D → int
    gt   = np.asarray(npz["GEO_TRANSFORM"]).ravel()  # ensure flat 1-D

    # pick any band to get H, W
    band_key = [k for k in npz.keys() if k not in ("SRID", "GEO_TRANSFORM", "VERSION")][0]
    h, w = npz[band_key].shape

    x_min = float(gt[0])
    y_max = float(gt[3])
    x_max = x_min + w * float(gt[1])
    y_min = y_max + h * float(gt[5])   # gt[5] is negative

    print(f"File  : {npz_path.name}")
    print(f"EPSG  : {srid}")
    print(f"Size  : {w} x {h} px  |  pixel size: {gt[1]:.4f} x {abs(gt[5]):.4f} units")
    print(f"BBox  : x [{x_min:.2f}, {x_max:.2f}]  y [{y_min:.2f}, {y_max:.2f}]")
    print(f"GeoTransform: {gt.tolist()}")
    return dict(epsg=srid, x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
                pixel_w=float(gt[1]), pixel_h=abs(float(gt[5])), width=w, height=h)

# Run on first few tiles
from src.dataset import load_tiles_json
from src.config import DATA_DIR

tiles = load_tiles_json()
for t in tiles[:3]:
    print()
    inspect_tile_geo(DATA_DIR / t["imagery_file"])

from src.config import DATA_DIR
from src.dataset import load_tiles_json
import numpy as np

epsg_counts = {}
for t in load_tiles_json()[:200]:   # sample 200
    npz = np.load(DATA_DIR / t["imagery_file"])
    e = int(np.asarray(npz["SRID"]).flat[0])
    epsg_counts[e] = epsg_counts.get(e, 0) + 1
print(epsg_counts)
