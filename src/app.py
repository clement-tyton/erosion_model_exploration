"""
Streamlit — Erosion model evaluation dashboard.

Tabs:
  1. Overview  — global KPIs + metric distributions
  2. Explorer  — filter/sort tiles, click row → overlay visualisation (on-the-fly)
  3. Data      — full parquet viewer + DuckDB SQL console

Run with:
    streamlit run src/app.py
"""

from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from src.config import OUTPUT_DIR, ROOT, TILES_DIR

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Erosion tile explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Model discovery ───────────────────────────────────────────────────────────
MODELS_DIR = ROOT / "models"
_model_files = sorted(MODELS_DIR.glob("*.pth")) if MODELS_DIR.exists() else []

if not _model_files:
    st.error(f"No .pth model files found in `{MODELS_DIR}`.")
    st.stop()

# ── Registry — maps model filename → dataset_name + tiles_json_id ─────────────
import json as _json_mod
_REGISTRY_PATH = ROOT / "models_registry.json"
_REGISTRY: dict[str, dict] = {}
if _REGISTRY_PATH.exists():
    for _e in _json_mod.loads(_REGISTRY_PATH.read_text()):
        _REGISTRY[_e["model_file"]] = _e


def _registry_entry(model_name: str) -> dict:
    return _REGISTRY.get(model_name, {})

# ── Metrics file lookup ───────────────────────────────────────────────────────
def _metrics_path(stem: str) -> Path | None:
    for p in [
        OUTPUT_DIR / f"metrics_{stem}.parquet",
        OUTPUT_DIR / f"metrics_{stem}.csv",
        OUTPUT_DIR / "metrics.parquet",
        OUTPUT_DIR / "metrics.csv",
    ]:
        if p.exists():
            return p
    return None


# ── Title row: "Erosion model —" + inline model selectbox ────────────────────
_t_col, _s_col = st.columns([1, 2])
with _t_col:
    st.markdown("### Erosion model —")
with _s_col:
    selected_model_name = st.selectbox(
        "model_select", [f.name for f in _model_files], index=0, label_visibility="collapsed"
    )

selected_model_path = MODELS_DIR / selected_model_name
model_stem = selected_model_path.stem
metrics_file = _metrics_path(model_stem)

if metrics_file is None:
    st.warning(
        f"No metrics for **{selected_model_name}** — run:  \n"
        f"`python -m src.evaluate --model-path models/{selected_model_name}`"
    )


# ── DuckDB — keyed by metrics file path ──────────────────────────────────────
@st.cache_resource
def get_con(path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    p = Path(path)
    if p.suffix == ".parquet":
        con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{p}')")
    else:
        con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_csv_auto('{p}')")
    return con


if metrics_file is None:
    st.stop()

con = get_con(str(metrics_file))


def q(query: str) -> pd.DataFrame:
    return con.execute(query).df()


# ── Model — CPU, keyed by path to support switching ──────────────────────────
@st.cache_resource(show_spinner="Loading model on CPU…")
def get_model(model_path: str):
    from src.model import load_model
    device = torch.device("cpu")
    model = load_model(Path(model_path), device=device)
    return model, device


# ── Tile JSON + data_dir — resolved from registry for the selected model ──────
_TILES_JSON_DIR = ROOT / "tiles_locations_json"


def _model_tiles_json(model_name: str) -> Path:
    """Local path to the tiles JSON manifest for this model."""
    entry = _registry_entry(model_name)
    if entry and "dataset_name" in entry:
        # preferred: tiles_locations_json/<dataset_name>.json
        p = _TILES_JSON_DIR / f"{entry['dataset_name']}.json"
        if p.exists():
            return p
        # legacy fallback: root level
        p = ROOT / f"{entry['dataset_name']}.json"
        if p.exists():
            return p
    if entry and "tiles_json_id" in entry:
        p = ROOT / f"{entry['tiles_json_id']}.json"
        if p.exists():
            return p
    from src.config import TILES_JSON
    return TILES_JSON


def _model_data_dir(model_name: str) -> Path:
    """Local directory containing NPZ tiles for this model."""
    entry = _registry_entry(model_name)
    if entry and "dataset_name" in entry:
        p = ROOT / "data" / entry["dataset_name"]
        if p.exists():
            return p
    # legacy fallback
    from src.config import DATA_DIR
    return DATA_DIR


@st.cache_data(show_spinner=False)
def tile_map(model_name: str) -> dict:
    from src.dataset import load_tiles_json
    return {t["imagery_file"]: t for t in load_tiles_json(_model_tiles_json(model_name))}


# ── PNG paths — subdirectory per model ───────────────────────────────────────
def _p(imagery_file: str, style: str) -> Path:
    stem = Path(imagery_file).stem
    suffix = "_overlay" if style == "overlay" else ""
    return TILES_DIR / model_stem / f"{stem}{suffix}.png"


def generate_pngs(imagery_file: str, metrics_row: dict) -> tuple[Path, Path]:
    from src.dataset import TileDataset
    from src.visualize import save_tile_overlay_png, save_tile_png

    p1 = _p(imagery_file, "masks")
    p2 = _p(imagery_file, "overlay")

    if p1.exists() and p2.exists():
        return p1, p2

    entry = tile_map(selected_model_name).get(imagery_file)
    if entry is None:
        return p1, p2

    model, device = get_model(str(selected_model_path))
    ds = TileDataset([entry], data_dir=_model_data_dir(selected_model_name))
    image, mask, _ = ds[0]

    # Pad to next multiple of 32 (SMP encoder requirement), then crop back
    _, h, w = image.shape
    def _pad32(x): return ((x + 31) // 32) * 32
    ph, pw = _pad32(h), _pad32(w)
    img_padded = torch.nn.functional.pad(image, (0, pw - w, 0, ph - h))

    with torch.no_grad():
        prob = model(img_padded.unsqueeze(0).to(device))
    pred = prob.argmax(dim=1).squeeze(0).cpu().numpy()[:h, :w]  # crop to original

    img_np = image.numpy()
    mask_np = mask.numpy()
    p1.parent.mkdir(parents=True, exist_ok=True)

    if not p1.exists():
        save_tile_png(
            imagery_file=imagery_file, img_chw=img_np,
            pred_mask=pred, true_mask=mask_np,
            metrics=metrics_row, out_path=p1,
        )
    if not p2.exists():
        save_tile_overlay_png(
            imagery_file=imagery_file, img_chw=img_np,
            pred_mask=pred, true_mask=mask_np,
            metrics=metrics_row, out_path=p2,
        )
    return p1, p2


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_explorer, tab_map, tab_compare, tab_data = st.tabs(
    ["Overview", "Tile explorer", "Map", "Compare models", "Raw data"]
)


# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab_overview:

    _has_cm = "tp_erosion" in con.execute("SELECT * FROM metrics LIMIT 0").df().columns

    kpi = q(f"""
        SELECT
            COUNT(*)                                                        AS n_tiles,
            COUNTIF(n_erosion_pixels > 0)                                   AS n_tiles_with_erosion,
            AVG(CASE WHEN n_erosion_pixels > 0 THEN f1_erosion        END)  AS f1_erosion_nonzero,
            AVG(CASE WHEN n_erosion_pixels > 0 THEN precision_erosion END)  AS precision_nonzero,
            AVG(CASE WHEN n_erosion_pixels > 0 THEN recall_erosion    END)  AS recall_nonzero,
            AVG(f1_no_erosion)                                              AS f1_no_erosion
            {"," if _has_cm else ""}
            {"SUM(tp_erosion) AS sum_tp, SUM(fp_erosion) AS sum_fp, SUM(fn_erosion) AS sum_fn" if _has_cm else ""}
        FROM metrics
    """).iloc[0]

    # Micro (global pixel-level) F1 — only if TP/FP/FN columns exist
    if _has_cm:
        _tp, _fp, _fn = float(kpi["sum_tp"]), float(kpi["sum_fp"]), float(kpi["sum_fn"])
        _denom = 2 * _tp + _fp + _fn
        global_f1 = (2 * _tp / _denom) if _denom > 0 else float("nan")
    else:
        global_f1 = None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tiles evaluated",      f"{int(kpi['n_tiles']):,}")
    c2.metric("Tiles with erosion",   f"{int(kpi['n_tiles_with_erosion']):,}",
              help="Ground-truth has ≥1 erosion pixel")
    c3.metric("Mean F1 erosion",      f"{kpi['f1_erosion_nonzero']:.4f}",
              help="Macro: average of per-tile F1, on erosion tiles only")
    if global_f1 is not None:
        c4.metric("Global F1 erosion",    f"{global_f1:.4f}",
                  help="Micro: 2·ΣTP / (2·ΣTP + ΣFP + ΣFN) across all pixels")
    else:
        c4.metric("Global F1 erosion",    "—",
                  help="Rerun evaluate.py with --force to compute (needs TP/FP/FN columns)")
    c5.metric("Recall (erosion)",     f"{kpi['recall_nonzero']:.4f}",
              help="Mean recall on erosion tiles")
    c6.metric("F1 no-erosion (ref.)", f"{kpi['f1_no_erosion']:.4f}")

    # ── Single slider ─────────────────────────────────────────────────────────
    max_px_val = int(q("SELECT MAX(n_erosion_pixels) FROM metrics").iloc[0, 0])
    min_px = st.slider(
        "Min erosion pixels — raise to exclude small/sparse erosion patches",
        min_value=0, max_value=max_px_val, value=0, step=50,
        key="overview_min_px",
    )

    # ── Dynamic KPIs (pale red) — erosion tiles only, filtered by slider ────────
    _eff_min = max(min_px, 1)   # always exclude zero-erosion tiles
    _cm_cols = (
        ", SUM(tp_erosion) AS sum_tp, SUM(fp_erosion) AS sum_fp, SUM(fn_erosion) AS sum_fn"
        if _has_cm else ""
    )
    dkpi = q(f"""
        SELECT
            COUNT(*)               AS n_tiles,
            AVG(f1_erosion)        AS f1,
            AVG(precision_erosion) AS prec,
            AVG(recall_erosion)    AS rec,
            AVG(f1_no_erosion)     AS f1_no
            {_cm_cols}
        FROM metrics
        WHERE n_erosion_pixels >= {_eff_min}
    """).iloc[0]

    if _has_cm:
        _dtp, _dfp, _dfn = float(dkpi["sum_tp"]), float(dkpi["sum_fp"]), float(dkpi["sum_fn"])
        _dd = 2 * _dtp + _dfp + _dfn
        d_global_f1 = (2 * _dtp / _dd) if _dd > 0 else float("nan")
    else:
        d_global_f1 = None

    def _fmt(v: object, dec: int = 4) -> str:
        try:
            f = float(v)
            return "N/A" if (f != f) else f"{f:.{dec}f}"   # NaN check
        except (TypeError, ValueError):
            return "N/A"

    def _red_metric(label: str, value: str) -> str:
        return (
            f'<div style="background:#fff0f0;border-radius:8px;padding:10px 4px;'
            f'text-align:center;margin:2px">'
            f'<div style="color:#888;font-size:0.78em;margin-bottom:4px">{label}</div>'
            f'<div style="color:#c0392b;font-size:1.5em;font-weight:700">{value}</div>'
            f'<div style="color:#bbb;font-size:0.72em">≥ {_eff_min:,} px</div>'
            f'</div>'
        )

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.markdown(_red_metric("Tiles (filtered)", f"{int(dkpi['n_tiles']):,}"),  unsafe_allow_html=True)
    r2.markdown(_red_metric("Mean F1 erosion",  _fmt(dkpi['f1'])),            unsafe_allow_html=True)
    r3.markdown(_red_metric("Global F1 erosion",
                            _fmt(d_global_f1) if d_global_f1 is not None else "—"),
                unsafe_allow_html=True)
    r4.markdown(_red_metric("Precision",        _fmt(dkpi['prec'])),          unsafe_allow_html=True)
    r5.markdown(_red_metric("Recall",           _fmt(dkpi['rec'])),           unsafe_allow_html=True)
    r6.markdown(_red_metric("F1 no-erosion",    _fmt(dkpi['f1_no'])),        unsafe_allow_html=True)

    st.divider()

    # ── Scatter ───────────────────────────────────────────────────────────────
    st.subheader("Precision vs Recall — erosion class")

    scatter = q(f"""
        SELECT imagery_file, precision_erosion, recall_erosion,
               f1_erosion, n_erosion_pixels
        FROM metrics
        WHERE n_erosion_pixels >= {min_px}
    """)
    cap = scatter["n_erosion_pixels"].quantile(0.95) if len(scatter) else 1
    scatter["_size"] = scatter["n_erosion_pixels"].clip(upper=cap) + 1

    fig_scatter = px.scatter(
        scatter,
        x="recall_erosion", y="precision_erosion",
        color="f1_erosion", size="_size",
        hover_data=["imagery_file", "f1_erosion", "n_erosion_pixels"],
        color_continuous_scale="RdYlGn",
        title=f"Each point = 1 tile · colour = F1 · size = erosion pixels · {len(scatter):,} tiles",
        labels={"recall_erosion": "Recall (erosion)", "precision_erosion": "Precision (erosion)"},
    )
    fig_scatter.update_layout(height=650)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Histograms ────────────────────────────────────────────────────────────
    st.subheader("Metric distributions")

    dist = q(f"""
        SELECT f1_erosion, precision_erosion, recall_erosion, f1_no_erosion
        FROM metrics
        WHERE n_erosion_pixels >= {min_px}
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(px.histogram(dist, x="f1_erosion", nbins=60,
            title="F1 erosion", color_discrete_sequence=["#e05252"]),
            use_container_width=True)
        st.plotly_chart(px.histogram(dist, x="recall_erosion", nbins=60,
            title="Recall erosion", color_discrete_sequence=["#5288e0"]),
            use_container_width=True)
    with col_b:
        st.plotly_chart(px.histogram(dist, x="precision_erosion", nbins=60,
            title="Precision erosion", color_discrete_sequence=["#52c4a0"]),
            use_container_width=True)
        st.plotly_chart(px.histogram(dist, x="f1_no_erosion", nbins=60,
            title="F1 no-erosion (reference)", color_discrete_sequence=["#9c52e0"]),
            use_container_width=True)

    st.divider()
    wc, bc = st.columns(2)
    with wc:
        st.subheader("Worst 10 — F1 erosion")
        st.dataframe(q("""
            SELECT imagery_file,
                   ROUND(f1_erosion, 4) AS f1_ero,
                   ROUND(precision_erosion, 4) AS prec,
                   ROUND(recall_erosion, 4) AS rec,
                   n_erosion_pixels AS ero_px
            FROM metrics WHERE n_erosion_pixels > 0
            ORDER BY f1_erosion ASC LIMIT 10
        """), width="stretch", hide_index=True)
    with bc:
        st.subheader("Best 10 — F1 erosion")
        st.dataframe(q("""
            SELECT imagery_file,
                   ROUND(f1_erosion, 4) AS f1_ero,
                   ROUND(precision_erosion, 4) AS prec,
                   ROUND(recall_erosion, 4) AS rec,
                   n_erosion_pixels AS ero_px
            FROM metrics ORDER BY f1_erosion DESC LIMIT 10
        """), width="stretch", hide_index=True)


# ── TAB 2: Tile explorer ──────────────────────────────────────────────────────
with tab_explorer:
    st.subheader("Tile explorer")

    # Sort options: simple (col, dir) or compound (raw ORDER BY clause)
    sort_opts = {
        "★ Worst F1 + most erosion px":    "f1_erosion ASC, n_erosion_pixels DESC",
        "★ Best F1 + most erosion px":     "f1_erosion DESC, n_erosion_pixels DESC",
        "F1 erosion ↑ (worst first)":      "f1_erosion ASC",
        "F1 erosion ↓ (best first)":       "f1_erosion DESC",
        "Precision ↑ (worst first)":       "precision_erosion ASC",
        "Recall ↑ (worst first)":          "recall_erosion ASC",
        "Most erosion pixels":             "n_erosion_pixels DESC",
        "Least erosion pixels":            "n_erosion_pixels ASC",
    }

    sec_opts = {
        "None":                    "",
        "Most erosion pixels":     ", n_erosion_pixels DESC",
        "Least erosion pixels":    ", n_erosion_pixels ASC",
        "F1 erosion ↑":            ", f1_erosion ASC",
        "F1 erosion ↓":            ", f1_erosion DESC",
        "Precision ↑":             ", precision_erosion ASC",
        "Recall ↑":                ", recall_erosion ASC",
    }

    fc1, fc2, fc3, fc4, fc5 = st.columns([2, 2, 2, 2, 1])
    with fc1:
        sort_label = st.selectbox("Sort by (primary)", list(sort_opts.keys()), index=0)
        order_clause = sort_opts[sort_label]
    with fc2:
        sec_label = st.selectbox("Then by (secondary)", list(sec_opts.keys()), index=0)
        order_clause += sec_opts[sec_label]
    with fc3:
        f1_range = st.slider("F1 erosion range", 0.0, 1.0, (0.0, 1.0), step=0.01)
    with fc4:
        max_px = int(q("SELECT MAX(n_erosion_pixels) FROM metrics").iloc[0, 0])
        exp_min_px = st.number_input("Min erosion pixels", 0, max_px, 200, step=100)
    with fc5:
        page_size = st.selectbox("Rows / page", [20, 50, 100], index=0)

    filtered = q(f"""
        SELECT imagery_file, mask_file,
               ROUND(f1_erosion, 4)        AS f1_erosion,
               ROUND(precision_erosion, 4) AS precision,
               ROUND(recall_erosion, 4)    AS recall,
               ROUND(iou_erosion, 4)       AS iou,
               n_erosion_pixels,
               n_no_erosion_pixels
        FROM metrics
        WHERE f1_erosion BETWEEN {f1_range[0]} AND {f1_range[1]}
          AND n_erosion_pixels >= {exp_min_px}
        ORDER BY {order_clause}
    """)

    total = len(filtered)
    total_pages = max(1, (total - 1) // page_size + 1)

    pc1, pc2 = st.columns([4, 1])
    with pc1:
        st.caption(f"**{total:,}** tiles match filters")
    with pc2:
        page_num = st.number_input("Page", 1, total_pages, 1, key="exp_page")

    page_df = filtered.iloc[(page_num - 1) * page_size : page_num * page_size]

    event = st.dataframe(
        page_df, use_container_width=True, hide_index=True,
        on_select="rerun", selection_mode="single-row", key="tile_table",
    )

    st.divider()

    sel_rows = event.selection.rows if event.selection else []
    if not sel_rows:
        st.info("Click a row to view the tile.")
    else:
        imagery_file = page_df.iloc[sel_rows[0]]["imagery_file"]
        mask_file    = page_df.iloc[sel_rows[0]]["mask_file"]

        st.subheader("Selected tile")

        nc1, nc2 = st.columns(2)
        with nc1:
            st.markdown("**Imagery file**")
            st.code(imagery_file, language=None)
        with nc2:
            st.markdown("**Mask file**")
            st.code(mask_file, language=None)

        full = q(f"""
            SELECT * FROM metrics WHERE imagery_file = '{imagery_file}' LIMIT 1
        """).iloc[0].to_dict()

        with st.spinner("Generating visualisation (cached after first run)…"):
            try:
                p1, p2 = generate_pngs(imagery_file, full)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                p1, p2 = _p(imagery_file, "masks"), _p(imagery_file, "overlay")

        visu_mode = st.radio(
            "Visualisation",
            options=["Side-by-side masks", "Contour overlay (RGB / DSM)"],
            horizontal=True, key="visu_mode",
        )
        _, img_col, _ = st.columns([2, 6, 2])
        with img_col:
            if visu_mode == "Side-by-side masks":
                if p1.exists():
                    st.image(str(p1), width="stretch")
                else:
                    st.warning("NPZ missing — cannot generate")
            else:
                if p2.exists():
                    st.image(str(p2), width="stretch")
                else:
                    st.warning("NPZ missing — cannot generate")

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("**Erosion**")
            st.dataframe(pd.DataFrame({
                "": ["F1", "Precision", "Recall", "IOU", "True pixels"],
                "Value": [
                    f"{full['f1_erosion']:.4f}", f"{full['precision_erosion']:.4f}",
                    f"{full['recall_erosion']:.4f}", f"{full['iou_erosion']:.4f}",
                    f"{int(full['n_erosion_pixels']):,}",
                ],
            }), hide_index=True, width="stretch")
        with mc2:
            st.markdown("**No-erosion**")
            st.dataframe(pd.DataFrame({
                "": ["F1", "Precision", "Recall", "IOU", "True pixels"],
                "Value": [
                    f"{full['f1_no_erosion']:.4f}", f"{full['precision_no_erosion']:.4f}",
                    f"{full['recall_no_erosion']:.4f}", f"{full['iou_no_erosion']:.4f}",
                    f"{int(full['n_no_erosion_pixels']):,}",
                ],
            }), hide_index=True, width="stretch")


# ── TAB 3: Map ───────────────────────────────────────────────────────────────
with tab_map:
    st.subheader("Tile map — Western Australia (EPSG 20350 → WGS84)")

    from src.build_geo import geo_parquet_path as _geo_path
    _entry      = _registry_entry(selected_model_name)
    _dataset    = _entry.get("dataset_name", "default") if _entry else "default"
    GEO_PARQUET = _geo_path(_dataset)

    if not GEO_PARQUET.exists():
        st.warning(
            f"Geographic index not built yet for dataset **{_dataset}**. Run:\n"
            f"```\npython -m src.run_all --dry-run\n```\n"
            f"then `python -m src.run_all` to build it automatically."
        )
    else:
        # ── Load & join geo + metrics ─────────────────────────────────────────
        @st.cache_data(show_spinner="Loading geo index…")
        def load_map_data(geo_path: str, metrics_path: str) -> pd.DataFrame:
            import duckdb as _ddb
            con = _ddb.connect()
            return con.execute(f"""
                SELECT
                    g.imagery_file, g.lat, g.lon,
                    g.pixel_size_m, g.width_px, g.height_px,
                    m.f1_erosion, m.precision_erosion, m.recall_erosion,
                    m.iou_erosion, m.f1_no_erosion,
                    m.n_erosion_pixels, m.n_no_erosion_pixels,
                    m.mask_file
                FROM read_parquet('{geo_path}') g
                JOIN read_parquet('{metrics_path}') m
                  ON g.imagery_file = m.imagery_file
            """).df()

        _map_metrics = _metrics_path(model_stem)
        if _map_metrics and _map_metrics.suffix == ".parquet":
            map_df_full = load_map_data(str(GEO_PARQUET), str(_map_metrics))
        elif _map_metrics and _map_metrics.suffix == ".csv":
            _tmp = OUTPUT_DIR / "_tmp_metrics.parquet"
            if not _tmp.exists():
                pd.read_csv(_map_metrics).to_parquet(_tmp, index=False)
            map_df_full = load_map_data(str(GEO_PARQUET), str(_tmp))
        else:
            st.error("No metrics file found for the selected model.")
            map_df_full = pd.DataFrame()

        if not map_df_full.empty:
            # ── Filters ───────────────────────────────────────────────────────
            mc1, mc2, mc3 = st.columns([2, 2, 2])
            with mc1:
                map_min_px = st.slider(
                    "Min erosion pixels",
                    0, int(map_df_full["n_erosion_pixels"].max()), 0, step=50,
                    key="map_min_px",
                )
            with mc2:
                map_f1_range = st.slider(
                    "F1 erosion range", 0.0, 1.0, (0.0, 1.0), step=0.01,
                    key="map_f1_range",
                )
            with mc3:
                _color_opts = [
                    "recall_erosion",
                    "f1_erosion",
                    "precision_erosion",
                    "n_erosion_pixels",
                    "pixel_size_m",
                ]
                color_metric = st.selectbox(
                    "Colour by", _color_opts, index=0, key="map_color",
                )

            map_df = map_df_full[
                (map_df_full["n_erosion_pixels"] >= map_min_px)
                & (map_df_full["f1_erosion"].between(*map_f1_range))
            ].copy()

            st.caption(f"**{len(map_df):,}** tiles shown")

            if map_df.empty:
                st.info("No tiles match the current filters.")
            else:
                cap = map_df["n_erosion_pixels"].quantile(0.95) + 1
                map_df["_size"] = map_df["n_erosion_pixels"].clip(upper=cap) + 1

                _center_lat = map_df["lat"].mean()
                _center_lon = map_df["lon"].mean()

                fig_map = px.scatter_map(
                    map_df,
                    lat="lat", lon="lon",
                    color=color_metric,
                    size="_size",
                    hover_name="imagery_file",
                    hover_data={
                        "f1_erosion": ":.3f",
                        "precision_erosion": ":.3f",
                        "recall_erosion": ":.3f",
                        "n_erosion_pixels": True,
                        "_size": False,
                        "lat": False,
                        "lon": False,
                    },
                    color_continuous_scale="RdYlGn",
                    center={"lat": _center_lat, "lon": _center_lon},
                    zoom=11,
                    height=650,
                    map_style="carto-darkmatter",
                    title=f"Tiles coloured by {color_metric} · {len(map_df):,} tiles",
                )
                fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})

                map_event = st.plotly_chart(
                    fig_map, use_container_width=True,
                    on_select="rerun", key="map_chart",
                )

                # ── Tile detail on click ───────────────────────────────────────
                pts = (map_event.selection.points
                       if map_event and map_event.selection else [])

                if pts:
                    clicked_file = pts[0].get("hovertext") or pts[0].get("customdata", [None])[0]
                    if clicked_file:
                        st.divider()
                        st.subheader(f"Selected: `{Path(clicked_file).stem}`")

                        nc1, nc2 = st.columns(2)
                        with nc1:
                            st.markdown("**Imagery file**")
                            st.code(clicked_file, language=None)

                        full_map = q(f"""
                            SELECT * FROM metrics
                            WHERE imagery_file = '{clicked_file}' LIMIT 1
                        """).iloc[0].to_dict()

                        with nc2:
                            st.markdown("**Mask file**")
                            st.code(full_map.get("mask_file", ""), language=None)

                        with st.spinner("Generating visualisation…"):
                            try:
                                p1, p2 = generate_pngs(clicked_file, full_map)
                            except Exception as e:
                                st.error(f"Generation failed: {e}")
                                p1, p2 = _p(clicked_file, "masks"), _p(clicked_file, "overlay")

                        visu_mode_map = st.radio(
                            "Visualisation",
                            ["Side-by-side masks", "Contour overlay (RGB / DSM)"],
                            horizontal=True, key="map_visu_mode",
                        )
                        _, img_col_map, _ = st.columns([2, 6, 2])
                        with img_col_map:
                            p = p1 if visu_mode_map == "Side-by-side masks" else p2
                            if p.exists():
                                st.image(str(p), width="stretch")
                            else:
                                st.warning("NPZ missing — cannot generate")

                        mm1, mm2 = st.columns(2)
                        with mm1:
                            st.markdown("**Erosion**")
                            st.dataframe(pd.DataFrame({
                                "": ["F1", "Precision", "Recall", "IOU", "True pixels"],
                                "Value": [
                                    f"{full_map['f1_erosion']:.4f}",
                                    f"{full_map['precision_erosion']:.4f}",
                                    f"{full_map['recall_erosion']:.4f}",
                                    f"{full_map['iou_erosion']:.4f}",
                                    f"{int(full_map['n_erosion_pixels']):,}",
                                ],
                            }), hide_index=True, width="stretch")
                        with mm2:
                            st.markdown("**No-erosion**")
                            st.dataframe(pd.DataFrame({
                                "": ["F1", "Precision", "Recall", "IOU", "True pixels"],
                                "Value": [
                                    f"{full_map['f1_no_erosion']:.4f}",
                                    f"{full_map['precision_no_erosion']:.4f}",
                                    f"{full_map['recall_no_erosion']:.4f}",
                                    f"{full_map['iou_no_erosion']:.4f}",
                                    f"{int(full_map['n_no_erosion_pixels']):,}",
                                ],
                            }), hide_index=True, width="stretch")


# ── TAB 4: Compare models ────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Model comparison")

    if not _REGISTRY_PATH.exists():
        st.warning("No `models_registry.json` found at repo root.")
    else:
        # ── Aggregate metrics (one row per model) ─────────────────────────────
        @st.cache_data(show_spinner="Loading aggregate metrics…")
        def _load_all_metrics(registry_hash: str) -> pd.DataFrame:
            rows = []
            for entry in list(_REGISTRY.values()):
                stem = Path(entry["model_file"]).stem
                pq   = next(
                    (p for p in [OUTPUT_DIR / f"metrics_{stem}.parquet",
                                 OUTPUT_DIR / "metrics.parquet"] if p.exists()),
                    None,
                )
                if pq is None:
                    continue

                import duckdb as _ddb
                _has_cm = "tp_erosion" in _ddb.connect().execute(
                    f"SELECT * FROM read_parquet('{pq}') LIMIT 0"
                ).df().columns
                _cm_expr = (
                    ", SUM(tp_erosion) AS sum_tp"
                    ", SUM(fp_erosion) AS sum_fp"
                    ", SUM(fn_erosion) AS sum_fn"
                    if _has_cm else ""
                )
                agg = _ddb.connect().execute(f"""
                    SELECT
                        COUNT(*)                                                        AS n_tiles,
                        COUNTIF(n_erosion_pixels > 0)                                  AS n_erosion_tiles,
                        SUM(n_erosion_pixels)                                          AS total_erosion_px,
                        SUM(n_no_erosion_pixels)                                       AS total_no_erosion_px,
                        AVG(CASE WHEN n_erosion_pixels > 0 THEN f1_erosion        END) AS mean_f1_erosion,
                        AVG(CASE WHEN n_erosion_pixels > 0 THEN precision_erosion END) AS mean_precision,
                        AVG(CASE WHEN n_erosion_pixels > 0 THEN recall_erosion    END) AS mean_recall,
                        AVG(f1_no_erosion)                                             AS mean_f1_no_erosion,
                        AVG(iou_erosion)                                               AS mean_iou_erosion
                        {_cm_expr}
                    FROM read_parquet('{pq}')
                """).df().iloc[0]

                _tp = float(agg.get("sum_tp", float("nan")))
                _fp = float(agg.get("sum_fp", float("nan")))
                _fn = float(agg.get("sum_fn", float("nan")))
                _d  = 2 * _tp + _fp + _fn
                global_f1 = (2 * _tp / _d) if _d > 0 else float("nan")

                total_px = float(agg["total_erosion_px"]) + float(agg["total_no_erosion_px"])
                rows.append({
                    "model":              entry["model_file"],
                    "label":              f"{Path(entry['model_file']).stem}",
                    "version":            entry["version"],
                    "epoch":              entry["epoch"],
                    "description":        entry["description"],
                    "dataset":            entry.get("dataset_name", entry["tiles_json_id"][:8]),
                    "n_tiles":            int(agg["n_tiles"]),
                    "n_erosion_tiles":    int(agg["n_erosion_tiles"]),
                    "erosion_prevalence": round(float(agg["total_erosion_px"]) / total_px * 100, 2) if total_px else float("nan"),
                    "mean_f1_erosion":    round(float(agg["mean_f1_erosion"]), 4),
                    "global_f1_erosion":  round(global_f1, 4),
                    "mean_precision":     round(float(agg["mean_precision"]), 4),
                    "mean_recall":        round(float(agg["mean_recall"]), 4),
                    "mean_f1_no_erosion": round(float(agg["mean_f1_no_erosion"]), 4),
                    "mean_iou_erosion":   round(float(agg["mean_iou_erosion"]), 4),
                })
            return pd.DataFrame(rows)

        # ── Full per-tile data (for distributions & pairwise) ─────────────────
        @st.cache_data(show_spinner="Loading tile-level data…")
        def _load_tile_data(registry_hash: str) -> dict[str, pd.DataFrame]:
            out = {}
            for entry in list(_REGISTRY.values()):
                stem = Path(entry["model_file"]).stem
                pq   = next(
                    (p for p in [OUTPUT_DIR / f"metrics_{stem}.parquet",
                                 OUTPUT_DIR / "metrics.parquet"] if p.exists()),
                    None,
                )
                if pq is None:
                    continue
                out[entry["model_file"]] = pd.read_parquet(pq)
            return out

        _reg_hash  = str(hash(_REGISTRY_PATH.read_text()))
        compare_df = _load_all_metrics(_reg_hash)
        tile_data  = _load_tile_data(_reg_hash)

        if compare_df.empty:
            st.info(
                "No metrics computed yet. Run:\n"
                "```\npython -m src.run_all --dry-run\n```\n"
                "to see what's missing, then run without `--dry-run` to evaluate."
            )
        else:
            n_evaluated = len(compare_df)
            n_total     = len(_REGISTRY)
            st.caption(
                f"**{n_evaluated} / {n_total}** models evaluated "
                f"({'run `python -m src.run_all` for the rest' if n_evaluated < n_total else 'all done ✅'})"
            )

            ranked = compare_df.sort_values("global_f1_erosion", ascending=False).reset_index(drop=True)
            winner = ranked.iloc[0]
            runner = ranked.iloc[1] if len(ranked) > 1 else None
            _gap   = (f"  (+{winner['global_f1_erosion'] - runner['global_f1_erosion']:.4f} vs runner-up)"
                      if runner is not None else "")
            st.success(
                f"🏆 **Winner: `{winner['label']}`** — "
                f"Global F1 erosion: **{winner['global_f1_erosion']:.4f}**{_gap}"
            )

            # ── Section 1: Leaderboard ────────────────────────────────────────
            st.markdown("#### 1 · Leaderboard")
            _metric_col_map = {
                "Mean F1 erosion (macro)":   "mean_f1_erosion",
                "Global F1 erosion (micro)": "global_f1_erosion",
                "Mean Recall":               "mean_recall",
                "Mean Precision":            "mean_precision",
                "Mean IOU erosion":          "mean_iou_erosion",
                "Mean F1 no-erosion":        "mean_f1_no_erosion",
            }
            _lb_sort = st.selectbox("Sort by", list(_metric_col_map.keys()),
                                    index=1, key="lb_sort")
            _lb_col  = _metric_col_map[_lb_sort]

            _lb_cols = [
                "label", "dataset", "epoch",
                "n_tiles", "n_erosion_tiles", "erosion_prevalence",
                "global_f1_erosion", "mean_f1_erosion",
                "mean_precision", "mean_recall",
                "mean_f1_no_erosion", "mean_iou_erosion",
            ]

            def _hi_max(s):
                return ["background-color:#1a472a;color:white" if v == s.max() else "" for v in s]

            _num_cols = ["global_f1_erosion", "mean_f1_erosion", "mean_precision",
                         "mean_recall", "mean_iou_erosion", "mean_f1_no_erosion"]
            styled_lb = (
                compare_df[_lb_cols]
                .sort_values(_lb_col, ascending=False)
                .style
                .apply(_hi_max, subset=_num_cols)
                .format({c: "{:.4f}" for c in _num_cols} | {"erosion_prevalence": "{:.2f}%"})
            )
            st.dataframe(styled_lb, hide_index=True, width="stretch")

            # Bar chart (all metrics)
            fig_bar = px.bar(
                compare_df.sort_values(_lb_col),
                x=_lb_col, y="label", orientation="h",
                color=_lb_col, color_continuous_scale="RdYlGn",
                text=_lb_col,
                title=_lb_sort,
                labels={_lb_col: _lb_sort, "label": ""},
            )
            fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_bar.update_layout(height=80 + 60 * len(compare_df),
                                  showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()

            # ── Section 2: Distribution comparison ───────────────────────────
            st.markdown("#### 2 · F1 distributions across tiles")
            st.caption("Shape of the distribution matters: a higher mean with a fat left tail means more catastrophic failures.")

            _d_filter = st.radio(
                "Tiles",
                ["All", "Erosion only (n_erosion_pixels > 0)"],
                horizontal=True, key="d_filter",
            )
            _d_models = st.multiselect(
                "Models", list(tile_data.keys()),
                default=list(tile_data.keys()), key="d_models",
                format_func=lambda m: Path(m).stem,
            )

            if _d_models:
                _dist_frames = []
                for _mf in _d_models:
                    _df = tile_data[_mf].copy()
                    if _d_filter.startswith("Erosion"):
                        _df = _df[_df["n_erosion_pixels"] > 0]
                    _df["model"] = Path(_mf).stem
                    _dist_frames.append(_df[["model", "f1_erosion", "f1_no_erosion", "iou_erosion"]])
                _dist_all = pd.concat(_dist_frames, ignore_index=True)

                _dc1, _dc2 = st.columns(2)
                with _dc1:
                    _fig_box = px.box(
                        _dist_all, x="model", y="f1_erosion", color="model",
                        title="F1 erosion — box plot",
                        labels={"f1_erosion": "F1 erosion", "model": ""},
                    )
                    _fig_box.update_layout(showlegend=False, height=420)
                    _fig_box.update_xaxes(tickangle=-25)
                    st.plotly_chart(_fig_box, use_container_width=True)

                with _dc2:
                    _fig_hist = px.histogram(
                        _dist_all, x="f1_erosion", color="model",
                        nbins=40, barmode="overlay", opacity=0.55,
                        title="F1 erosion — histogram",
                        labels={"f1_erosion": "F1 erosion", "count": "# tiles"},
                    )
                    _fig_hist.update_layout(height=420)
                    st.plotly_chart(_fig_hist, use_container_width=True)

            st.divider()

            # ── Section 3: Pairwise deep-dive ─────────────────────────────────
            st.markdown("#### 3 · Pairwise deep-dive")
            st.caption("Join on shared tiles — works best when models share a dataset. "
                       "Points above the diagonal → B is better; below → A is better.")

            _model_list = list(tile_data.keys())
            _pc1, _pc2, _pc3 = st.columns([3, 3, 2])
            with _pc1:
                _pw_a = st.selectbox("Model A", _model_list, index=0, key="pw_a",
                                     format_func=lambda m: Path(m).stem)
            with _pc2:
                _pw_b = st.selectbox("Model B", _model_list,
                                     index=min(1, len(_model_list) - 1), key="pw_b",
                                     format_func=lambda m: Path(m).stem)
            with _pc3:
                _win_thresh = st.slider("Win threshold (Δ F1)", 0.01, 0.20, 0.05,
                                        step=0.01, key="pw_thresh")

            if _pw_a == _pw_b:
                st.warning("Select two different models.")
            else:
                _cols_pw = ["imagery_file", "f1_erosion", "precision_erosion",
                            "recall_erosion", "iou_erosion", "n_erosion_pixels"]
                _df_a = tile_data[_pw_a][_cols_pw].copy()
                _df_b = tile_data[_pw_b][_cols_pw].copy()
                _merged = _df_a.merge(_df_b, on="imagery_file", suffixes=("_a", "_b"))
                _n_common = len(_merged)

                if _n_common == 0:
                    st.warning("No tiles in common — these models used different datasets.")
                else:
                    _stem_a = Path(_pw_a).stem
                    _stem_b = Path(_pw_b).stem
                    st.caption(f"**{_n_common:,}** tiles in common")

                    _merged["delta"] = _merged["f1_erosion_a"] - _merged["f1_erosion_b"]
                    _merged["result"] = _merged["delta"].apply(
                        lambda d: f"A wins ({_stem_a})" if d > _win_thresh
                                  else (f"B wins ({_stem_b})" if d < -_win_thresh else "Tie")
                    )

                    _n_a   = (_merged["delta"] >  _win_thresh).sum()
                    _n_b   = (_merged["delta"] < -_win_thresh).sum()
                    _n_tie = _n_common - _n_a - _n_b

                    _kc1, _kc2, _kc3 = st.columns(3)
                    _kc1.metric(f"A wins — {_stem_a}", int(_n_a))
                    _kc2.metric("Ties", int(_n_tie))
                    _kc3.metric(f"B wins — {_stem_b}", int(_n_b))

                    _fig_pw = px.scatter(
                        _merged,
                        x="f1_erosion_a", y="f1_erosion_b",
                        color="result",
                        color_discrete_map={
                            f"A wins ({_stem_a})": "#2196F3",
                            f"B wins ({_stem_b})": "#FF5722",
                            "Tie": "#888888",
                        },
                        size="n_erosion_pixels_a",
                        size_max=14,
                        hover_data={"imagery_file": True, "delta": ":.4f",
                                    "n_erosion_pixels_a": True},
                        title=f"{_stem_a}  vs  {_stem_b} — tile-level F1 erosion",
                        labels={
                            "f1_erosion_a": f"F1 erosion — A ({_stem_a})",
                            "f1_erosion_b": f"F1 erosion — B ({_stem_b})",
                        },
                    )
                    _fig_pw.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                      line=dict(dash="dot", color="white", width=1))
                    _fig_pw.update_layout(height=560)
                    st.plotly_chart(_fig_pw, use_container_width=True)

                    _tc1, _tc2 = st.columns(2)
                    with _tc1:
                        st.markdown(f"**Tiles where A dominates** (Δ > {_win_thresh})")
                        _top_a = (
                            _merged[_merged["delta"] > _win_thresh]
                            .sort_values("delta", ascending=False)
                            [["imagery_file", "f1_erosion_a", "f1_erosion_b", "delta"]]
                            .rename(columns={"f1_erosion_a": "F1_A", "f1_erosion_b": "F1_B"})
                            .head(20)
                        )
                        st.dataframe(_top_a, hide_index=True, width="stretch")

                    with _tc2:
                        st.markdown(f"**Tiles where B dominates** (Δ < -{_win_thresh})")
                        _top_b = (
                            _merged[_merged["delta"] < -_win_thresh]
                            .sort_values("delta", ascending=True)
                            [["imagery_file", "f1_erosion_a", "f1_erosion_b", "delta"]]
                            .rename(columns={"f1_erosion_a": "F1_A", "f1_erosion_b": "F1_B"})
                            .head(20)
                        )
                        st.dataframe(_top_b, hide_index=True, width="stretch")

            st.divider()

            # ── Section 4: Performance by erosion density ─────────────────────
            st.markdown("#### 4 · Performance by erosion density")
            st.caption("Which model handles sparse erosion (hard) vs dense erosion (easy) better?")

            _dens_frames = []
            for _mf, _df_m in tile_data.items():
                _dc = _df_m.copy()
                _total = (_dc["n_erosion_pixels"] + _dc["n_no_erosion_pixels"]).clip(lower=1)
                _pct   = _dc["n_erosion_pixels"] / _total * 100
                _dc["density"] = pd.cut(
                    _pct,
                    bins=[-0.001, 0, 5, 25, 100],
                    labels=["None (0%)", "Sparse (0–5%)", "Medium (5–25%)", "Dense (>25%)"],
                )
                _dc["model"] = Path(_mf).stem
                _dens_frames.append(_dc[["model", "density", "f1_erosion"]])

            if _dens_frames:
                _dens_all = pd.concat(_dens_frames, ignore_index=True)
                _dens_agg = (
                    _dens_all.groupby(["model", "density"], observed=True)["f1_erosion"]
                    .agg(mean_f1="mean", n="count").reset_index()
                )
                _fig_dens = px.bar(
                    _dens_agg,
                    x="density", y="mean_f1", color="model",
                    barmode="group", text_auto=".3f",
                    title="Mean F1 erosion by erosion density bucket",
                    labels={"density": "Erosion density", "mean_f1": "Mean F1 erosion"},
                    category_orders={"density": ["None (0%)", "Sparse (0–5%)",
                                                  "Medium (5–25%)", "Dense (>25%)"]},
                    hover_data={"n": True},
                )
                _fig_dens.update_layout(height=440)
                st.plotly_chart(_fig_dens, use_container_width=True)


# ── TAB 5: Raw data ───────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Raw data")

    st.markdown("**Parquet — first 5 000 rows (sorted by F1 erosion ↑)**")
    st.dataframe(
        q("SELECT * FROM metrics ORDER BY f1_erosion ASC LIMIT 5000"),
        width="stretch", hide_index=True, height=420,
    )

    st.divider()
    st.markdown("**DuckDB SQL console**")
    st.caption("Query the `metrics` view. Example: `SELECT * FROM metrics WHERE f1_erosion < 0.1 LIMIT 20`")

    user_sql = st.text_area(
        "SQL",
        value="SELECT * FROM metrics ORDER BY f1_erosion ASC LIMIT 20",
        height=160,
    )
    if st.button("Run"):
        try:
            res = q(user_sql)
            st.success(f"{len(res):,} rows")
            st.dataframe(res, width="stretch", hide_index=True)
        except Exception as e:
            st.error(str(e))
