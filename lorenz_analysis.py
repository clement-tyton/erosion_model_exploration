"""
Lorenz-style analysis of erosion pixel distribution and model performance
across tiles sorted by ascending erosion pixel count.

Usage:
    python lorenz_analysis.py
    python lorenz_analysis.py --parquet output/metrics_model_v3_split_test_epoch80.parquet
    python lorenz_analysis.py --parquet output/metrics_model_v3_split_test_epoch80.parquet --out lorenz.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DEFAULT_PARQUET = "output/metrics_model_v3_split_test_epoch_1819.parquet"


def load_and_prepare(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    # Keep only tiles that actually contain erosion pixels
    df = df[df["n_erosion_pixels"] > 0].copy()
    # Sort by ascending erosion pixel count (poorest → richest, Lorenz convention)
    df = df.sort_values("n_erosion_pixels", ascending=True).reset_index(drop=True)
    return df


def compute_curves(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    x = (np.arange(1, n + 1) / n) * 100  # percent of tiles

    # --- Lorenz: cumulative share of erosion pixels ---
    total_erosion = df["n_erosion_pixels"].sum()
    lorenz_y = df["n_erosion_pixels"].cumsum() / total_erosion * 100

    # --- Global precision / recall / F1 on the first k tiles ---
    cum_tp = df["tp_erosion"].cumsum()
    cum_fp = df["fp_erosion"].cumsum()
    cum_fn = df["fn_erosion"].cumsum()

    precision = cum_tp / (cum_tp + cum_fp).replace(0, np.nan)
    recall    = cum_tp / (cum_tp + cum_fn).replace(0, np.nan)
    f1        = 2 * precision * recall / (precision + recall).replace(0, np.nan)

    return pd.DataFrame({
        "x_pct_tiles":  x,
        "lorenz_y":     lorenz_y.values,
        "precision":    precision.values,
        "recall":       recall.values,
        "f1":           f1.values,
    })


def annotate_lorenz(ax, curves: pd.DataFrame):
    """Add the classic '10% of tiles → X% of erosion' annotation."""
    for tile_pct in [10, 20, 50]:
        idx = np.searchsorted(curves["x_pct_tiles"], tile_pct)
        if idx >= len(curves):
            continue
        erosion_pct = curves["lorenz_y"].iloc[idx]
        ax.annotate(
            f"{tile_pct}% tiles\n→ {erosion_pct:.1f}% erosion",
            xy=(tile_pct, erosion_pct),
            xytext=(tile_pct + 3, erosion_pct - 8),
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            color="gray",
        )


def plot_lorenz(curves: pd.DataFrame, ax: plt.Axes, total_erosion_tiles: int):
    x = curves["x_pct_tiles"]
    y = curves["lorenz_y"]

    ax.fill_between(x, y, x, alpha=0.15, color="steelblue", label="Inequality area")
    ax.plot(x, y, color="steelblue", lw=2, label="Lorenz curve")
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="Perfect equality")

    annotate_lorenz(ax, curves)

    ax.set_title(f"Lorenz Curve – Erosion Pixel Distribution\n(n = {total_erosion_tiles:,} tiles with erosion)", fontsize=10)
    ax.set_xlabel("Cumulative % of tiles\n(sorted by ascending erosion pixel count)")
    ax.set_ylabel("Cumulative % of total erosion pixels")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_metric(curves: pd.DataFrame, ax: plt.Axes, metric: str, color: str):
    x = curves["x_pct_tiles"]
    y = curves[metric] * 100  # convert 0-1 → percent

    ax.plot(x, y, color=color, lw=2)
    ax.axhline(y.iloc[-1], color=color, lw=0.8, linestyle=":", alpha=0.7,
               label=f"Overall: {y.iloc[-1]:.1f}%")

    # Annotate where metric starts degrading significantly (first cross below 50%)
    below50 = np.where(y.values < 50)[0]
    if len(below50):
        idx = below50[0]
        ax.annotate(
            f"<50% at {x.iloc[idx]:.1f}%\nof tiles",
            xy=(x.iloc[idx], y.iloc[idx]),
            xytext=(x.iloc[idx] + 5, y.iloc[idx] + 5),
            fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            color="gray",
        )

    labels = {"precision": "Precision (erosion)", "recall": "Recall (erosion)", "f1": "F1 score (erosion)"}
    ax.set_title(f"Global {labels[metric]}\n(computed on first N tiles)", fontsize=10)
    ax.set_xlabel("Cumulative % of tiles\n(sorted by ascending erosion pixel count)")
    ax.set_ylabel(labels[metric])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Lorenz-style erosion analysis")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET, help="Path to metrics parquet file")
    parser.add_argument("--out", default=None, help="Save figure to this path (optional)")
    args = parser.parse_args()

    print(f"Loading {args.parquet} …")
    df = load_and_prepare(args.parquet)
    print(f"  Tiles with erosion : {len(df):,}")
    print(f"  Total erosion px   : {df['n_erosion_pixels'].sum():,}")

    curves = compute_curves(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Erosion Tile Analysis — {args.parquet.split('/')[-1]}",
        fontsize=12, fontweight="bold",
    )

    plot_lorenz(curves, axes[0, 0], len(df))
    plot_metric(curves, axes[0, 1], "precision", color="darkorange")
    plot_metric(curves, axes[1, 0], "recall",    color="seagreen")
    plot_metric(curves, axes[1, 1], "f1",        color="mediumvioletred")

    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
