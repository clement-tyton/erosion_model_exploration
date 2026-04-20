"""
Tab 1 — Statistics

Lorenz-style analysis of erosion distribution and model performance inequality
across tiles sorted by ascending erosion pixel count.

Sections
--------
1. Lorenz curve         — cumulative % of erosion pixels vs % of tiles
2. Cumulative precision — global precision on the first N (worst→best) tiles
3. Cumulative recall    — global recall
4. Cumulative F1        — global F1 score
5. Spatial autocorrelation — empirical variogram + recommended tile size
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ---------------------------------------------------------------------------
# Section 1-4 — Lorenz + cumulative curves
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Computing Lorenz curves…")
def _compute_curves(metrics_path: str) -> pd.DataFrame:
    df = pd.read_parquet(metrics_path)
    df = df[df["n_erosion_pixels"] > 0].copy()
    df = df.sort_values("n_erosion_pixels", ascending=True).reset_index(drop=True)

    n = len(df)
    if n == 0:
        return pd.DataFrame()

    x = (np.arange(1, n + 1) / n) * 100

    # Lorenz
    total_erosion = df["n_erosion_pixels"].sum()
    lorenz_y = df["n_erosion_pixels"].cumsum() / total_erosion * 100

    # Cumulative global precision / recall / F1
    cum_tp = df["tp_erosion"].cumsum()
    cum_fp = df["fp_erosion"].cumsum()
    cum_fn = df["fn_erosion"].cumsum()
    denom_p = (cum_tp + cum_fp).replace(0, np.nan)
    denom_r = (cum_tp + cum_fn).replace(0, np.nan)
    precision = cum_tp / denom_p * 100
    recall    = cum_tp / denom_r * 100
    f1        = 2 * precision * recall / (precision + recall).replace(0, np.nan)

    return pd.DataFrame({
        "x":         x,
        "lorenz_y":  lorenz_y.values,
        "precision": precision.values,
        "recall":    recall.values,
        "f1":        f1.values,
        "n_erosion_pixels": df["n_erosion_pixels"].values,
    })


# ---------------------------------------------------------------------------
# Plot helpers — unified 2×2 subplot (ggplot-like tight layout)
# ---------------------------------------------------------------------------

def _combined_fig(
    curves: pd.DataFrame,
    model_stem: str,
    n_tiles: int,
    overall_prec: float,
    overall_recall: float,
    overall_f1: float,
) -> go.Figure:
    """Single Plotly figure with 4 tight subplots — much closer to ggplot facets."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Lorenz curve · {n_tiles:,} tiles with erosion",
            "Global precision (erosion)",
            "Global recall (erosion)",
            "Global F1 score (erosion)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.09,
    )

    x = curves["x"]

    # ── (1,1) Lorenz ──────────────────────────────────────────────────────────
    # Diagonal first so fill="tonexty" fills between diagonal and curve
    fig.add_trace(go.Scatter(
        x=[0, 100], y=[0, 100], mode="lines",
        line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=curves["lorenz_y"], mode="lines", name="Lorenz",
        line=dict(color="#4a9eda", width=2.5),
        fill="tonexty", fillcolor="rgba(74,158,218,0.11)",
        hovertemplate="%{x:.1f}% tiles → %{y:.1f}% erosion<extra></extra>",
    ), row=1, col=1)

    for pct in [10, 20, 50]:
        idx = int(np.searchsorted(x, pct))
        if idx < len(curves):
            ero = float(curves["lorenz_y"].iloc[idx])
            fig.add_annotation(
                x=pct, y=ero, xref="x", yref="y",
                text=f"{pct}%→{ero:.0f}%",
                showarrow=True, arrowhead=2, arrowcolor="#777",
                font=dict(size=8, color="#aaa"), ax=26, ay=-18,
            )

    # ── (1,2) Precision ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=curves["precision"], mode="lines", name="Precision",
        line=dict(color="#f0983a", width=2.5),
        hovertemplate="%{x:.1f}% tiles → prec %{y:.1f}%<extra></extra>",
    ), row=1, col=2)
    fig.add_hline(y=overall_prec * 100, line_dash="dot", line_color="#f0983a",
                  line_width=1.2, row=1, col=2,
                  annotation_text=f"  {overall_prec * 100:.1f}%",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#f0983a"))

    # ── (2,1) Recall ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=curves["recall"], mode="lines", name="Recall",
        line=dict(color="#3acf7a", width=2.5),
        hovertemplate="%{x:.1f}% tiles → rec %{y:.1f}%<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=overall_recall * 100, line_dash="dot", line_color="#3acf7a",
                  line_width=1.2, row=2, col=1,
                  annotation_text=f"  {overall_recall * 100:.1f}%",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#3acf7a"))

    # ── (2,2) F1 ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=curves["f1"], mode="lines", name="F1",
        line=dict(color="#c85ab4", width=2.5),
        hovertemplate="%{x:.1f}% tiles → F1 %{y:.1f}%<extra></extra>",
    ), row=2, col=2)
    fig.add_hline(y=overall_f1 * 100, line_dash="dot", line_color="#c85ab4",
                  line_width=1.2, row=2, col=2,
                  annotation_text=f"  {overall_f1 * 100:.1f}%",
                  annotation_position="bottom right",
                  annotation_font=dict(size=9, color="#c85ab4"))

    # ── Shared axis formatting ─────────────────────────────────────────────────
    _grid = dict(gridcolor="rgba(255,255,255,0.07)", gridwidth=1)
    for r, c, yrange in [(1,1,[0,100]), (1,2,[0,105]), (2,1,[0,105]), (2,2,[0,105])]:
        fig.update_xaxes(
            range=[0, 100], ticksuffix="%", title_text="% of tiles (sorted by erosion)",
            title_font_size=10, tickfont_size=9, row=r, col=c, **_grid,
        )
        fig.update_yaxes(
            range=yrange, ticksuffix="%",
            tickfont_size=9, row=r, col=c, **_grid,
        )

    fig.update_layout(
        height=740,
        showlegend=False,
        margin=dict(l=52, r=22, t=50, b=38),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11),
    )
    # Subtitle font size
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return fig


# ---------------------------------------------------------------------------
# Section 5 — Spatial autocorrelation / variogram
# ---------------------------------------------------------------------------

_CLUSTER_EPS = 200.0   # metres  — DBSCAN radius to group tiles into sites
_MAX_DIST    = 150.0   # metres  — max variogram lag
_N_LAGS      = 30
_MIN_PAIRS   = 30      # minimum pairs per lag bin
_GSD_MEDIAN  = 0.021   # m/px   — median GSD for this UAV dataset (~2.1 cm/px)
_TILE_PX_STD = 384     # standard tile side in pixels


def _find_geo_path(model_stem: str) -> Path | None:
    """Heuristic: find the tiles_geo_*.parquet that best matches the model stem."""
    from src.config import OUTPUT_DIR
    stem_no_model = model_stem.removeprefix("model_")
    base = stem_no_model.split("_epoch")[0]   # e.g. "v3_split_test"
    # Sort by file size descending so if multiple match we take the most complete one
    candidates = sorted(
        OUTPUT_DIR.glob("tiles_geo_*.parquet"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    for c in candidates:
        # Skip the test geo file
        if c.stem == "tiles_geo_test":
            continue
        if base in c.stem:
            return c
    return None


def _spherical_model(h: np.ndarray, nugget: float, sill: float, range_: float) -> np.ndarray:
    hr = h / range_
    return np.where(
        h <= range_,
        nugget + sill * (1.5 * hr - 0.5 * hr ** 3),
        nugget + sill,
    )


@st.cache_data(show_spinner="Computing spatial autocorrelation… (first run ~30 s)")
def _compute_variogram_data(metrics_path: str, geo_path: str) -> dict | None:
    from scipy.optimize import curve_fit
    from scipy.spatial.distance import pdist
    from sklearn.cluster import DBSCAN

    metrics = pd.read_parquet(metrics_path)[["imagery_file", "n_erosion_pixels"]]
    geo     = pd.read_parquet(geo_path)[["imagery_file", "x_center", "y_center",
                                          "pixel_size_m", "width_px"]]
    df = metrics.merge(geo, on="imagery_file", how="inner")
    if df.empty:
        return None

    df["tile_size_m"] = df["pixel_size_m"] * df["width_px"]
    tile_size_m = float(df["tile_size_m"].median())

    # ── DBSCAN geographic clustering ──────────────────────────────────────────
    coords = df[["x_center", "y_center"]].values
    labels = DBSCAN(eps=_CLUSTER_EPS, min_samples=2, algorithm="ball_tree",
                    metric="euclidean").fit_predict(coords)
    df["site_id"] = labels
    df = df[df["site_id"] >= 0].reset_index(drop=True)
    if df.empty:
        return None

    n_sites = df["site_id"].nunique()

    # ── Empirical variogram — within-site pairs only ──────────────────────────
    lag_edges    = np.linspace(0, _MAX_DIST, _N_LAGS + 1)
    lag_centres  = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    semivar      = np.full(_N_LAGS, np.nan)
    n_pairs_arr  = np.zeros(_N_LAGS, dtype=int)

    for site_id in df["site_id"].unique():
        sub = df[df["site_id"] == site_id].reset_index(drop=True)
        if len(sub) < 2:
            continue
        c      = sub[["x_center", "y_center"]].values
        z      = sub["n_erosion_pixels"].values.astype(float)
        dists  = pdist(c)
        sqdiff = pdist(z[:, None], metric="sqeuclidean")
        mask_w = dists <= _MAX_DIST
        if mask_w.sum() == 0:
            continue
        dw, sw = dists[mask_w], sqdiff[mask_w]
        for k in range(_N_LAGS):
            m       = (dw >= lag_edges[k]) & (dw < lag_edges[k + 1])
            curr_n  = int(m.sum())
            if curr_n < _MIN_PAIRS:
                continue
            sv      = 0.5 * float(sw[m].mean())
            prev_n  = n_pairs_arr[k]
            if np.isnan(semivar[k]):
                semivar[k]    = sv
                n_pairs_arr[k] = curr_n
            else:
                total          = prev_n + curr_n
                semivar[k]     = (semivar[k] * prev_n + sv * curr_n) / total
                n_pairs_arr[k] = total

    vario = pd.DataFrame({
        "lag_m":   lag_centres,
        "gamma":   semivar,
        "n_pairs": n_pairs_arr,
    }).dropna(subset=["gamma"])

    if len(vario) < 3:
        return None

    # ── Spherical model fit ───────────────────────────────────────────────────
    h, g = vario["lag_m"].values, vario["gamma"].values
    nugget = sill = range_ = None
    try:
        popt, _ = curve_fit(
            _spherical_model, h, g,
            p0=[float(np.nanmin(g)),
                float(np.nanmax(g) - np.nanmin(g)),
                float(h[np.nanargmax(g)] / 2)],
            bounds=([0, 0, 1], [np.inf, np.inf, np.inf]),
            maxfev=10_000,
        )
        nugget, sill, range_ = float(popt[0]), float(popt[1]), float(popt[2])
    except RuntimeError:
        pass

    sites_df = df[["x_center", "y_center", "site_id"]].copy()

    return {
        "vario":        vario,
        "nugget":       nugget,
        "sill":         sill,
        "range_":       range_,
        "tile_size_m":  tile_size_m,
        "n_sites":      n_sites,
        "n_tiles_geo":  len(df),
        "sites_df":     sites_df,
    }


def _variogram_fig(result: dict) -> go.Figure:
    vario       = result["vario"]
    nugget      = result["nugget"]
    sill        = result["sill"]
    range_      = result["range_"]
    tile_size_m = result["tile_size_m"]

    # ── Transform to √γ — "semi-deviation" in px ─────────────────────────────
    sqrt_gamma = np.sqrt(vario["gamma"].values)

    fig = go.Figure()

    sizes = np.clip(np.log1p(vario["n_pairs"].values) * 2.0, 4, 13)
    fig.add_trace(go.Scatter(
        x=vario["lag_m"], y=sqrt_gamma,
        mode="markers",
        marker=dict(color="#4a9eda", size=sizes.tolist(), opacity=0.9,
                    line=dict(color="white", width=0.5)),
        name="√γ(h) empirical  (size ∝ n pairs)",
        hovertemplate=(
            "h = %{x:.1f} m<br>"
            "√γ = %{y:.0f} px<br>"
            "n pairs: %{text}<extra></extra>"
        ),
        text=vario["n_pairs"].tolist(),
    ))

    if range_ is not None:
        h_fit      = np.linspace(0, float(vario["lag_m"].max()), 500)
        g_fit      = _spherical_model(h_fit, nugget, sill, range_)
        sqrt_g_fit = np.sqrt(g_fit)
        fig.add_trace(go.Scatter(
            x=h_fit, y=sqrt_g_fit, mode="lines",
            line=dict(color="#e05252", width=2.2),
            name=f"Spherical fit — range = {range_:.1f} m",
        ))
        fig.add_vline(
            x=range_,
            line_dash="dash", line_color="#e05252", line_width=1.5,
            annotation_text=f"<b>range = {range_:.1f} m</b>",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#e05252"),
        )
        fig.add_hline(
            y=np.sqrt(nugget + sill),
            line_dash="dot", line_color="gray", line_width=1,
            annotation_text=f"√sill = {np.sqrt(nugget + sill):.0f} px",
            annotation_position="right",
            annotation_font=dict(size=9, color="gray"),
        )
        fig.add_hline(
            y=np.sqrt(nugget),
            line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1,
            annotation_text=f"√nugget = {np.sqrt(nugget):.0f} px",
            annotation_position="right",
            annotation_font=dict(size=9, color="gray"),
        )

    fig.add_vline(
        x=tile_size_m,
        line_dash="dashdot", line_color="#f0983a", line_width=2,
        annotation_text=f"tile = {tile_size_m:.1f} m",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#f0983a"),
    )

    if range_ is not None and range_ > tile_size_m:
        fig.add_vrect(
            x0=tile_size_m, x1=range_,
            fillcolor="rgba(240,152,58,0.07)",
            line_width=0,
            annotation_text="gap → recall deficit",
            annotation_position="inside top",
            annotation_font=dict(size=9, color="#aaa"),
        )

    fig.update_layout(
        title="Empirical variogram — spatial autocorrelation of erosion",
        title_font_size=13,
        xaxis_title="Distance between tile centres (m)",
        yaxis_title="√γ(h)  [px]  —  semi-deviation of erosion count",
        height=400, width=620,
        legend=dict(orientation="h", y=1.11, font_size=10),
        margin=dict(l=75, r=25, t=60, b=50),
    )
    return fig


def _sites_fig(result: dict) -> go.Figure:
    """Scatter map of DBSCAN geographic clusters, coloured by site_id."""
    import plotly.express as px

    sites_df = result["sites_df"]
    n_sites  = result["n_sites"]

    # ── Single trace: all tiles, colour = site_id (continuous Turbo scale) ────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sites_df["x_center"],
        y=sites_df["y_center"],
        mode="markers",
        marker=dict(
            color=sites_df["site_id"],
            colorscale="Turbo",
            size=2,
            opacity=0.55,
            showscale=False,
        ),
        hovertemplate="Site %{text}<extra></extra>",
        text=sites_df["site_id"].tolist(),
        showlegend=False,
    ))

    # ── Centroid labels ───────────────────────────────────────────────────────
    centroids = sites_df.groupby("site_id")[["x_center", "y_center"]].mean()
    palette   = px.colors.qualitative.Alphabet   # 26 colours, cycled
    for i, (sid, row) in enumerate(centroids.iterrows()):
        col = palette[int(sid) % len(palette)]
        fig.add_annotation(
            x=row["x_center"], y=row["y_center"],
            text=str(sid),
            showarrow=False,
            font=dict(size=6, color="white", family="monospace"),
            bgcolor=col,
            borderpad=1,
            opacity=0.85,
        )

    fig.update_layout(
        title=f"DBSCAN clusters — {n_sites} sites  (ε = {_CLUSTER_EPS:.0f} m)",
        title_font_size=12,
        xaxis=dict(title=None, showticklabels=False,
                   showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title=None, showticklabels=False,
                   showgrid=True, gridcolor="rgba(255,255,255,0.08)",
                   scaleanchor="x", scaleratio=1),
        height=400, width=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(metrics_file: Path, model_stem: str) -> None:
    st.subheader("Erosion inequality & cumulative performance curves")
    st.caption(
        "Tiles are sorted by **ascending erosion pixel count** (poorest → richest). "
        "Tiles with zero erosion pixels are excluded from this analysis.  \n"
        "The Lorenz curve shows how unequally erosion is distributed across tiles. "
        "The three performance curves show how global precision/recall/F1 evolve as "
        "progressively more erosion-dense tiles are included in the evaluation."
    )

    curves = _compute_curves(str(metrics_file))
    if curves.empty:
        st.warning("No tiles with erosion pixels found in this metrics file.")
        return

    n_tiles        = len(curves)
    total_erosion  = int(curves["n_erosion_pixels"].sum())
    overall_prec   = float(curves["precision"].iloc[-1]) / 100
    overall_recall = float(curves["recall"].iloc[-1]) / 100
    overall_f1     = float(curves["f1"].iloc[-1]) / 100

    # ── KPI strip ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tiles with erosion",    f"{n_tiles:,}")
    k2.metric("Total erosion pixels",  f"{total_erosion:,}")
    k3.metric("Overall Precision",     f"{overall_prec:.4f}")
    k4.metric("Overall Recall",        f"{overall_recall:.4f}")
    k5.metric("Overall F1",            f"{overall_f1:.4f}")

    st.divider()
    st.caption(
        "At each point N the curves show the **global** (micro, pixel-level) metric "
        "computed over the first N tiles only. "
        "Precision = ΣTP / (ΣTP + ΣFP) · Recall = ΣTP / (ΣTP + ΣFN) · "
        "F1 = harmonic mean of the above two."
    )

    # ── Unified 2×2 subplot — single Plotly figure ────────────────────────────
    st.plotly_chart(
        _combined_fig(curves, model_stem, n_tiles, overall_prec, overall_recall, overall_f1),
        use_container_width=True,
    )

    st.info(
        "**Key insight:**  \n"
        "- **Precision** rises quickly — even among tiles with few erosion pixels, what "
        "the model labels as erosion is usually correct.  \n"
        "- **Recall** rises slowly and near-linearly — the model detects a higher "
        "fraction of erosion only once tiles are erosion-dense. This is the binding "
        "constraint.  \n"
        "- The **Lorenz curve** quantifies how concentrated erosion is: if 20% of tiles "
        "hold 80%+ of all erosion, boundary/fringe tiles dominate the recall deficit."
    )

    # =========================================================================
    # Section 5 — Spatial autocorrelation
    # =========================================================================
    st.divider()
    st.subheader("Spatial autocorrelation of erosion pixels")
    st.caption(
        "Empirical variogram of `n_erosion_pixels` across tile pairs "
        "(within geographic sites, DBSCAN ε = 200 m). "
        "The **range** of the fitted spherical model is the distance beyond which "
        "two tiles are statistically independent — i.e. the characteristic diameter "
        "of an erosion patch. Comparing it to the current tile footprint gives a "
        "principled recommendation for the optimal tile size."
    )

    geo_path = _find_geo_path(model_stem)
    if geo_path is None or not geo_path.exists():
        st.info(
            "No geo parquet found for this model. "
            "Run `python src/build_geo.py` to generate tile coordinates."
        )
        return

    result = _compute_variogram_data(str(metrics_file), str(geo_path))
    if result is None:
        st.warning("Could not compute variogram — insufficient tile pairs after filtering.")
        return

    range_      = result["range_"]
    tile_size_m = result["tile_size_m"]
    n_sites     = result["n_sites"]
    n_tiles_geo = result["n_tiles_geo"]

    # ── KPI strip ─────────────────────────────────────────────────────────────
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Geographic sites (DBSCAN)", f"{n_sites}")
    v2.metric("Tiles with coordinates", f"{n_tiles_geo:,}")
    v3.metric("Tile footprint (median)", f"{tile_size_m:.1f} m")
    if range_ is not None:
        rec_px = int(np.ceil((range_ / _GSD_MEDIAN) / _TILE_PX_STD) * _TILE_PX_STD)
        v4.metric("Autocorrelation range", f"{range_:.1f} m")
        v5.metric("Recommended tile size", f"{rec_px} px  ({rec_px * _GSD_MEDIAN:.1f} m)")
    else:
        v4.metric("Autocorrelation range", "fit failed")
        v5.metric("Recommended tile size", "—")

    # ── Variogram + site map side by side ─────────────────────────────────────
    col_vario, col_sites = st.columns([5, 3])
    with col_vario:
        st.plotly_chart(_variogram_fig(result), use_container_width=False)
    with col_sites:
        st.plotly_chart(_sites_fig(result), use_container_width=False)

    # ── Variogram interpretation ──────────────────────────────────────────────
    if range_ is not None:
        sqrt_nugget = float(np.sqrt(result["nugget"]))
        sqrt_total  = float(np.sqrt(result["nugget"] + result["sill"]))
        sqrt_sill   = float(np.sqrt(result["sill"]))
        noise_pct   = result["nugget"] / (result["nugget"] + result["sill"]) * 100
        struct_pct  = 100 - noise_pct

        st.markdown(
            f"**Reading the variogram** · y-axis = √γ(h), the **standard deviation of the "
            f"difference** in erosion pixel count between two tiles at distance h.  \n"
            f"- At h → 0: √γ ≈ **{sqrt_nugget:.0f} px** (nugget — irreducible local noise: "
            f"two adjacent tiles still differ by ~{sqrt_nugget:.0f} px on average, "
            f"regardless of how close they are).  \n"
            f"- The curve rises because tiles become more dissimilar as distance increases, "
            f"then flattens at **√γ = {sqrt_total:.0f} px** (total semi-deviation) once "
            f"tiles are fully independent.  \n"
            f"- That plateau is reached at **range = {range_:.1f} m**: beyond this distance "
            f"knowing one tile's erosion count tells you nothing about its neighbour's.  \n"
            f"- **{struct_pct:.0f}% of variance is spatially structured** (sill / total = "
            f"{sqrt_sill:.0f} / {sqrt_total:.0f} px): erosion is not random — patches have "
            f"a real spatial signature up to {range_:.1f} m.  \n"
            f"- The orange line marks the current tile footprint ({tile_size_m:.1f} m) — "
            f"it sits well inside the range. Most erosion-containing tiles therefore capture "
            f"only a fragment of a {range_:.1f} m patch, which directly explains why recall "
            f"collapses on boundary tiles."
        )

    # ── Tile size derivation ──────────────────────────────────────────────────
    if range_ is not None:
        px_needed = range_ / _GSD_MEDIAN
        rec_px    = int(np.ceil(px_needed / _TILE_PX_STD) * _TILE_PX_STD)
        factor    = rec_px / _TILE_PX_STD
        st.info(
            f"**Tile size derivation:**  \n"
            f"Range = **{range_:.1f} m** — current tile = **{tile_size_m:.1f} m** "
            f"(ratio **{range_ / tile_size_m:.1f}×**).  \n"
            f"At median GSD {_GSD_MEDIAN * 100:.1f} cm/px:  "
            f"`{range_:.1f} m ÷ {_GSD_MEDIAN * 100:.1f} cm/px = {px_needed:.0f} px` "
            f"→ rounded to **{rec_px} px** (= {factor:.0f}× the current {_TILE_PX_STD} px tile).  \n"
            f"Verification: `{rec_px} px × {_GSD_MEDIAN * 100:.1f} cm/px = "
            f"{rec_px * _GSD_MEDIAN:.1f} m ≥ {range_:.1f} m` ✓  \n"
            f"A tile of **{rec_px} × {rec_px} px** gives the model a field of view "
            f"that covers at least one complete erosion patch, which should "
            f"disproportionately improve recall on the currently-weak boundary tiles."
        )
