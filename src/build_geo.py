"""
Extract geographic centroids from tile NPZ files and save to
output/tiles_geo_<dataset_name>.parquet  (one file per dataset).

GeoTransform stored as a 3×3 affine matrix (9 values):
  [scale_x,  shear_x,  x_origin,
   shear_y,  scale_y,  y_origin,
   0,        0,        1        ]

All tiles are EPSG 20350 (GDA94 / MGA zone 50, Western Australia).
Centroids are reprojected to WGS84 (lat/lon) for web mapping.

Run for all datasets via run_all, or manually:
    python -m src.build_geo --dataset-name v3_split_test_final
    python -m src.build_geo   # defaults to the config TILES_JSON / DATA_DIR
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from tqdm import tqdm

from src.config import DATA_DIR, OUTPUT_DIR
from src.dataset import load_tiles_json

# Single transformer — all tiles are MGA zone 50
_TO_WGS84 = Transformer.from_crs("EPSG:20350", "EPSG:4326", always_xy=True)


def geo_parquet_path(dataset_name: str) -> Path:
    return OUTPUT_DIR / f"tiles_geo_{dataset_name}.parquet"


def _process_tile(entry: dict) -> dict | None:
    path = DATA_DIR / entry["imagery_file"]
    if not path.exists():
        return None
    try:
        npz = np.load(path, mmap_mode="r")

        srid = int(np.asarray(npz["SRID"]).flat[0])
        gt   = np.asarray(npz["GEO_TRANSFORM"]).ravel()

        # GeoTransform layout: row-major 3×3 affine
        # gt[0]=scale_x  gt[1]=shear_x  gt[2]=x_origin
        # gt[3]=shear_y  gt[4]=scale_y  gt[5]=y_origin
        x_origin = float(gt[2])
        y_origin = float(gt[5])
        pixel_w  = float(gt[0])
        pixel_h  = float(gt[4])   # negative (north-up)

        band_key = next(k for k in npz.keys()
                        if k not in ("SRID", "GEO_TRANSFORM", "VERSION"))
        h, w = npz[band_key].shape

        # Centroid in native CRS
        x_c = x_origin + (w / 2) * pixel_w
        y_c = y_origin + (h / 2) * pixel_h

        # Bounding box in native CRS
        x_min = x_origin
        x_max = x_origin + w * pixel_w
        y_max = y_origin                   # top-left (north)
        y_min = y_origin + h * pixel_h    # bottom (south, pixel_h < 0)

        # Reproject centroid → WGS84
        lon, lat = _TO_WGS84.transform(x_c, y_c)

        # Reproject bbox corners → WGS84
        lon_min, lat_min = _TO_WGS84.transform(x_min, y_min)
        lon_max, lat_max = _TO_WGS84.transform(x_max, y_max)

        return {
            "imagery_file": entry["imagery_file"],
            "epsg":         srid,
            "pixel_size_m": abs(pixel_w),
            "width_px":     w,
            "height_px":    h,
            "x_origin":     x_origin,
            "y_origin":     y_origin,
            "x_center":     x_c,
            "y_center":     y_c,
            "lat":          lat,
            "lon":          lon,
            "lat_min":      lat_min,
            "lat_max":      lat_max,
            "lon_min":      lon_min,
            "lon_max":      lon_max,
        }
    except Exception as e:
        print(f"[WARN] {entry['imagery_file']}: {e}")
        return None


def build_geo(
    dataset_name: str = "default",
    tiles_json: Path | None = None,
    data_dir: Path | None = None,
    workers: int = 8,
    force: bool = False,
) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out = geo_parquet_path(dataset_name)
    if out.exists() and not force:
        print(f"[build_geo] {out.name} already exists — skipping (use --force to rebuild).")
        return pd.read_parquet(out)

    _tiles_json = tiles_json or None  # load_tiles_json handles None → default
    _data_dir   = data_dir or DATA_DIR

    tiles = load_tiles_json(_tiles_json) if _tiles_json else load_tiles_json()

    # Patch DATA_DIR used inside _process_tile via closure
    def _proc(entry: dict) -> dict | None:
        path = _data_dir / entry["imagery_file"]
        if not path.exists():
            return None
        try:
            npz = np.load(path, mmap_mode="r")
            srid = int(np.asarray(npz["SRID"]).flat[0])
            gt   = np.asarray(npz["GEO_TRANSFORM"]).ravel()
            x_origin = float(gt[2]); y_origin = float(gt[5])
            pixel_w  = float(gt[0]); pixel_h  = float(gt[4])
            band_key = next(k for k in npz.keys()
                            if k not in ("SRID", "GEO_TRANSFORM", "VERSION"))
            h, w = npz[band_key].shape
            x_c = x_origin + (w / 2) * pixel_w
            y_c = y_origin + (h / 2) * pixel_h
            x_min = x_origin; x_max = x_origin + w * pixel_w
            y_max = y_origin; y_min = y_origin + h * pixel_h
            lon, lat = _TO_WGS84.transform(x_c, y_c)
            lon_min, lat_min = _TO_WGS84.transform(x_min, y_min)
            lon_max, lat_max = _TO_WGS84.transform(x_max, y_max)
            return {
                "imagery_file": entry["imagery_file"],
                "epsg":         srid,
                "pixel_size_m": abs(pixel_w),
                "width_px":     w,   "height_px":    h,
                "x_origin":     x_origin, "y_origin": y_origin,
                "x_center":     x_c,  "y_center":    y_c,
                "lat":          lat,  "lon":          lon,
                "lat_min":      lat_min, "lat_max":   lat_max,
                "lon_min":      lon_min, "lon_max":   lon_max,
            }
        except Exception as e:
            print(f"[WARN] {entry['imagery_file']}: {e}")
            return None

    print(f"[build_geo] {dataset_name}: {len(tiles):,} tiles, {workers} workers…")
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_proc, t): t for t in tiles}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="tile"):
            result = fut.result()
            if result:
                rows.append(result)

    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    print(f"[build_geo] Saved {len(df):,} rows → {out}")
    print(f"  Lat range : {df['lat'].min():.4f} → {df['lat'].max():.4f}")
    print(f"  Lon range : {df['lon'].min():.4f} → {df['lon'].max():.4f}")
    print(f"  Pixel sizes: {sorted(df['pixel_size_m'].unique())} m")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="default",
                        help="Dataset name (used in output filename)")
    parser.add_argument("--tiles-json", type=str, default=None)
    parser.add_argument("--data-dir",   type=str, default=None)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()
    build_geo(
        dataset_name=args.dataset_name,
        tiles_json=Path(args.tiles_json) if args.tiles_json else None,
        data_dir=Path(args.data_dir)     if args.data_dir   else None,
        workers=args.workers,
        force=args.force,
    )
