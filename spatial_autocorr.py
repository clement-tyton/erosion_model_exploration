"""
Empirical variogram of n_erosion_pixels across tiles to estimate the
spatial autocorrelation radius of erosion.

Steps:
  1. Merge metrics parquet (n_erosion_pixels) + geo parquet (x_center, y_center)
  2. Cluster tiles into geographic sites with DBSCAN (no UUID needed)
  3. Bin into lag classes, compute semi-variance γ(h) per lag — within sites only
  4. Fit a spherical model to find the range (autocorrelation radius)
  5. Plot variogram + site map

Usage:
    python spatial_autocorr.py
    python spatial_autocorr.py --cluster-eps 200 --max-dist 150 --n-lags 30
    python spatial_autocorr.py --show-sites          # show site map only
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

DEFAULT_METRICS     = "output/metrics_model_v3_split_test_epoch_1819.parquet"
DEFAULT_GEO         = "output/tiles_geo_v3_split_test_final.parquet"
DEFAULT_CLUSTER_EPS = 200   # metres — max gap inside a site
DEFAULT_MAX_DIST    = 150   # metres — max variogram lag (stay well within a site)
DEFAULT_N_LAGS      = 30


# ---------------------------------------------------------------------------
# Data loading + geographic clustering
# ---------------------------------------------------------------------------

def build_dataset(metrics_path: str, geo_path: str, cluster_eps: float) -> pd.DataFrame:
    metrics = pd.read_parquet(metrics_path)[["imagery_file", "n_erosion_pixels",
                                             "tp_erosion", "fp_erosion", "fn_erosion"]]
    geo     = pd.read_parquet(geo_path)[["imagery_file", "x_center", "y_center",
                                         "pixel_size_m", "width_px"]]
    df = metrics.merge(geo, on="imagery_file", how="inner")
    df["tile_size_m"] = df["pixel_size_m"] * df["width_px"]

    # ── Geographic clustering with DBSCAN ─────────────────────────────────────
    # eps = max distance between two tiles to be considered neighbours in the same site
    # min_samples=2 so even a pair of adjacent tiles forms a cluster
    coords = df[["x_center", "y_center"]].values
    labels = DBSCAN(eps=cluster_eps, min_samples=2, algorithm="ball_tree",
                    metric="euclidean").fit_predict(coords)
    df["site_id"] = labels   # -1 = isolated tile (no neighbour within eps)

    n_sites = (df["site_id"] >= 0).sum()
    n_noise = (df["site_id"] == -1).sum()
    n_clusters = df["site_id"].nunique() - (1 if -1 in df["site_id"].values else 0)
    print(f"  DBSCAN (eps={cluster_eps} m): {n_clusters} sites | "
          f"{n_sites:,} clustered tiles | {n_noise} isolated tiles (excluded)")

    return df[df["site_id"] >= 0].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Variogram computation
# ---------------------------------------------------------------------------

def compute_variogram(
    df: pd.DataFrame,
    max_dist_m: float,
    n_lags: int,
    min_pairs: int = 30,
) -> pd.DataFrame:
    """
    Empirical semi-variance γ(h) per distance lag.
    Pairs formed only within the same geographic site.
    """
    lag_edges   = np.linspace(0, max_dist_m, n_lags + 1)
    lag_centres = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    semivar     = np.full(n_lags, np.nan)
    n_pairs     = np.zeros(n_lags, dtype=int)

    sites = df["site_id"].unique()
    print(f"  Computing pairs across {len(sites)} sites …")

    for site_id in sites:
        sub = df[df["site_id"] == site_id].reset_index(drop=True)
        if len(sub) < 2:
            continue

        coords   = sub[["x_center", "y_center"]].values
        z        = sub["n_erosion_pixels"].values.astype(float)
        dists    = pdist(coords)
        sq_diffs = pdist(z[:, None], metric="sqeuclidean")  # (z_i - z_j)^2

        # Only keep pairs within max_dist (avoid crossing the site boundary)
        within = dists <= max_dist_m
        if within.sum() == 0:
            continue

        dists_w    = dists[within]
        sq_diffs_w = sq_diffs[within]

        for k in range(n_lags):
            mask    = (dists_w >= lag_edges[k]) & (dists_w < lag_edges[k + 1])
            curr_n  = int(mask.sum())
            if curr_n < min_pairs:
                continue
            sv = 0.5 * float(sq_diffs_w[mask].mean())

            prev_n = n_pairs[k]
            if np.isnan(semivar[k]):
                semivar[k] = sv
                n_pairs[k] = curr_n
            else:
                total      = prev_n + curr_n
                semivar[k] = (semivar[k] * prev_n + sv * curr_n) / total
                n_pairs[k] = total

    return pd.DataFrame({
        "lag_m":   lag_centres,
        "gamma":   semivar,
        "n_pairs": n_pairs,
    }).dropna(subset=["gamma"])


# ---------------------------------------------------------------------------
# Spherical model
# ---------------------------------------------------------------------------

def spherical_model(h, nugget, sill, range_):
    hr = h / range_
    return np.where(
        h <= range_,
        nugget + sill * (1.5 * hr - 0.5 * hr ** 3),
        nugget + sill,
    )


def fit_spherical(vario: pd.DataFrame):
    h, g = vario["lag_m"].values, vario["gamma"].values
    try:
        popt, _ = curve_fit(
            spherical_model, h, g,
            p0=[float(np.nanmin(g)),
                float(np.nanmax(g) - np.nanmin(g)),
                float(h[np.nanargmax(g)] / 2)],
            bounds=([0, 0, 1], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
        return tuple(popt)
    except RuntimeError:
        warnings.warn("Spherical model fit failed — showing raw variogram only.")
        return None, None, None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_sites(df: pd.DataFrame, cluster_eps: float, out: str | None = None):
    """Scatter map coloured by site_id so you can verify the clustering."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sites   = df["site_id"].unique()
    cmap    = cm.get_cmap("tab20", len(sites))

    for i, sid in enumerate(sorted(sites)):
        sub = df[df["site_id"] == sid]
        ax.scatter(sub["x_center"], sub["y_center"],
                   s=2, color=cmap(i % 20), alpha=0.6, rasterized=True)
        # label centroid with site id
        cx, cy = sub["x_center"].mean(), sub["y_center"].mean()
        ax.text(cx, cy, str(sid), fontsize=6, ha="center", va="center",
                color="black", fontweight="bold")

    ax.set_title(f"Geographic Sites — DBSCAN eps={cluster_eps} m  "
                 f"({len(sites)} sites, {len(df):,} tiles)", fontsize=11)
    ax.set_xlabel("x_center (m, EPSG 20350)")
    ax.set_ylabel("y_center (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Site map saved to {out}")
    else:
        plt.show()


def plot_variogram(vario: pd.DataFrame, nugget, sill, range_,
                   tile_size_m: float, out: str | None):
    fig, ax = plt.subplots(figsize=(9, 5))

    sizes = np.clip(np.log1p(vario["n_pairs"]) * 4, 10, 80)
    ax.scatter(vario["lag_m"], vario["gamma"], s=sizes, color="steelblue",
               zorder=3, label="Empirical γ(h)\n(size ∝ log n_pairs)")

    if range_ is not None:
        h_fit = np.linspace(0, vario["lag_m"].max(), 500)
        g_fit = spherical_model(h_fit, nugget, sill, range_)
        ax.plot(h_fit, g_fit, color="firebrick", lw=2,
                label=f"Spherical fit  (range = {range_:.1f} m)")
        ax.axvline(range_, color="firebrick", lw=1.2, linestyle="--", alpha=0.7)
        ax.axhline(nugget + sill, color="gray", lw=0.8, linestyle=":",
                   label=f"Sill = {nugget + sill:.2e}")
        ax.text(range_ + 0.5, vario["gamma"].min() * 0.98,
                f"range = {range_:.1f} m", color="firebrick", fontsize=9, va="top")

    ax.axvline(tile_size_m, color="darkorange", lw=1.5, linestyle="-.",
               label=f"Tile footprint ({tile_size_m:.1f} m)")

    ax.set_xlabel("Distance between tile centres (m)")
    ax.set_ylabel("Semi-variance  γ(h)  of  n_erosion_pixels")
    ax.set_title("Empirical Variogram — Spatial Autocorrelation of Erosion\n"
                 "(within-site pairs only, geographic DBSCAN clusters)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Variogram saved to {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics",     default=DEFAULT_METRICS)
    parser.add_argument("--geo",         default=DEFAULT_GEO)
    parser.add_argument("--cluster-eps", type=float, default=DEFAULT_CLUSTER_EPS,
                        help="DBSCAN radius (m) to group tiles into sites (default: 200)")
    parser.add_argument("--max-dist",    type=float, default=DEFAULT_MAX_DIST,
                        help="Max variogram lag in metres (default: 150)")
    parser.add_argument("--n-lags",      type=int,   default=DEFAULT_N_LAGS,
                        help="Number of distance bins (default: 30)")
    parser.add_argument("--show-sites",  action="store_true",
                        help="Show site map and exit")
    parser.add_argument("--out",         default=None)
    args = parser.parse_args()

    print("Loading data …")
    df = build_dataset(args.metrics, args.geo, args.cluster_eps)
    tile_size_m = float(df["tile_size_m"].median())
    print(f"  Median tile footprint : {tile_size_m:.1f} m")
    print(f"  Site sizes: min={df['site_id'].value_counts().min()} "
          f"median={df['site_id'].value_counts().median():.0f} "
          f"max={df['site_id'].value_counts().max()}")

    if args.show_sites:
        plot_sites(df, args.cluster_eps, args.out)
        return

    print(f"Computing variogram (max_dist={args.max_dist} m, {args.n_lags} lags) …")
    vario = compute_variogram(df, max_dist_m=args.max_dist, n_lags=args.n_lags)
    print(f"  Valid lag bins: {len(vario)}")
    print(vario.to_string(index=False))

    print("Fitting spherical model …")
    nugget, sill, range_ = fit_spherical(vario)
    if range_ is not None:
        print(f"  Nugget : {nugget:.2e}")
        print(f"  Sill   : {sill:.2e}")
        print(f"  Range  : {range_:.1f} m  ← autocorrelation radius")
        if range_ > tile_size_m:
            print(f"  ✓ Range ({range_:.1f} m) > tile ({tile_size_m:.1f} m) → bigger tiles should help recall")
        else:
            print(f"  ✗ Range ({range_:.1f} m) ≤ tile ({tile_size_m:.1f} m) → tile size not the bottleneck")

    plot_variogram(vario, nugget, sill, range_, tile_size_m, args.out)


# ---------------------------------------------------------------------------
# Debug / step-by-step version
# ---------------------------------------------------------------------------

def test():
    """
    Hardcoded step-by-step version for IDE debugging.
    Run with:  python spatial_autocorr.py --test
    """
    # ── 0. Config ─────────────────────────────────────────────────────────────
    metrics_path = "output/metrics_model_v3_split_test_epoch_1819.parquet"
    geo_path     = "output/tiles_geo_v3_split_test_final.parquet"
    cluster_eps  = 200    # metres — tune this to split sites correctly
    max_dist_m   = 150    # metres — variogram lag limit
    n_lags       = 30
    _out         = None   # set to a path string to save figures instead of showing

    # ── 1. Load raw parquets ───────────────────────────────────────────────────
    _metrics_raw = pd.read_parquet(metrics_path)
    _geo_raw     = pd.read_parquet(geo_path)
    # breakpoint: _metrics_raw.columns, _geo_raw.columns

    # ── 2. Merge + DBSCAN site clustering ─────────────────────────────────────
    df = build_dataset(metrics_path, geo_path, cluster_eps)
    # breakpoint: df["site_id"].value_counts(), df.head()

    tile_size_m  = float(df["tile_size_m"].median())
    _site_counts = df["site_id"].value_counts()
    # breakpoint: tile_size_m, _site_counts

    # ── 3. Inspect one site ────────────────────────────────────────────────────
    example_site   = _site_counts.index[0]   # largest site
    site_df        = df[df["site_id"] == example_site].reset_index(drop=True)
    site_coords    = site_df[["x_center", "y_center"]].values
    site_z         = site_df["n_erosion_pixels"].values.astype(float)
    _site_dists    = pdist(site_coords)
    _site_sqdiffs  = pdist(site_z[:, None], metric="sqeuclidean")
    # breakpoint: _site_dists.min(), _site_dists.max(), site_df.shape
    # → these are the pairs that feed the variogram for this site

    # ── 4. Compute full variogram ──────────────────────────────────────────────
    vario = compute_variogram(df, max_dist_m=max_dist_m, n_lags=n_lags)
    # breakpoint: vario

    _lag_edges   = np.linspace(0, max_dist_m, n_lags + 1)
    _lag_centres = 0.5 * (_lag_edges[:-1] + _lag_edges[1:])
    # breakpoint: _lag_edges, _lag_centres

    # ── 5. Fit spherical model ─────────────────────────────────────────────────
    nugget, sill, range_ = fit_spherical(vario)
    # breakpoint: nugget, sill, range_

    _h_fit = np.linspace(0, vario["lag_m"].max(), 500)
    _g_fit = spherical_model(_h_fit, nugget, sill, range_)
    # breakpoint: _g_fit[:10]

    # ── 6. Plot site map ──────────────────────────────────────────────────────
    plot_sites(df, cluster_eps, out=None)

    # ── 7. Plot variogram ─────────────────────────────────────────────────────
    plot_variogram(vario, nugget, sill, range_, tile_size_m, out=None)

    plt.show()   # blocks until you close both windows


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        test()
    else:
        main()
