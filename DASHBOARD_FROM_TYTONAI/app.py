"""
Erosion model evaluation dashboard.

Run with:
    streamlit run DASHBOARD_FROM_TYTONAI/app.py

Tabs
----
1. Overview      — global KPIs + metric distributions (training set)
2. Tile explorer — filter/sort training tiles, click row → visualisation
3. Map           — geospatial tiles coloured by metric
                   + purple test-set overlay (toggle)
4. Compare models — leaderboard with train + test columns, distributions,
                    pairwise deep-dive, density buckets
5. Test set      — test tile explorer with click-to-visualise
6. Raw data      — Parquet viewer + DuckDB SQL console
"""

from __future__ import annotations

import json as _json_mod
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# ── Project root & config ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
from src.config import OUTPUT_DIR, TILES_DIR

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Erosion tile explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model discovery ───────────────────────────────────────────────────────────
MODELS_DIR   = ROOT / "models"

# Display order: best/most-recent models first
_MODEL_DISPLAY_ORDER = [
    "model_v3_split_test_epoch400.pth",
    "model_v3_split_test_epoch399.pth",
    "model_v3_split_test_epoch_243.pth",
    "model_v3_split_test_epoch_95.pth",
    "model_v3_split_test_epoch80.pth",
    "model_v3_split_test_epoch78.pth",
    "model_v3_split_test_epoch50.pth",
    "model_v2_no_erosion_td_epoch50.pth",
    "model_v1_jaswinder_epoch50.pth",
    "model_finetuned_tytonai_epoch5.pth",
]

def _model_sort_key(p: Path) -> int:
    try:
        return _MODEL_DISPLAY_ORDER.index(p.name)
    except ValueError:
        return len(_MODEL_DISPLAY_ORDER)  # unknown models go last

_model_files = (
    sorted(MODELS_DIR.glob("*.pth"), key=_model_sort_key)
    if MODELS_DIR.exists() else []
)

if not _model_files:
    st.error(f"No .pth model files found in `{MODELS_DIR}`.")
    st.stop()

# ── Registry ──────────────────────────────────────────────────────────────────
_REGISTRY_PATH = ROOT / "models_registry.json"
_REGISTRY: dict[str, dict] = {}
if _REGISTRY_PATH.exists():
    for _e in _json_mod.loads(_REGISTRY_PATH.read_text()):
        _REGISTRY[_e["model_file"]] = _e


def _registry_entry(model_name: str) -> dict:
    return _REGISTRY.get(model_name, {})


# ── Train metrics path ────────────────────────────────────────────────────────
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


# ── Test metrics path ─────────────────────────────────────────────────────────
def _test_metrics_path(stem: str) -> Path | None:
    p = OUTPUT_DIR / f"test_metrics_{stem}.parquet"
    return p if p.exists() else None


def _test_geo_path() -> Path:
    return OUTPUT_DIR / "tiles_geo_test.parquet"


# ── Test data dir & metadata ───────────────────────────────────────────────────
_TEST_DATA_DIR    = ROOT / "Experiments_MLFLOW" / "data" / "test_data"
_TEST_METADATA_JSON = ROOT / "Experiments_MLFLOW" / "metadata" / "EROSION_DATASET_TEST_METADATA.json"

# ── Title row: "Erosion model —" + inline model selectbox ────────────────────
_t_col, _s_col = st.columns([1, 2])
with _t_col:
    st.markdown("### Erosion model —")
with _s_col:
    selected_model_name = st.selectbox(
        "model_select", [f.name for f in _model_files], index=0,
        label_visibility="collapsed",
    )

selected_model_path = MODELS_DIR / selected_model_name
model_stem          = selected_model_path.stem
metrics_file        = _metrics_path(model_stem)

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


# ── Model on CPU — keyed by path ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model on CPU…")
def get_model(model_path: str):
    from src.model import load_model
    device = torch.device("cpu")
    model = load_model(Path(model_path), device=device)
    return model, device


# ── Tile JSON + data_dir — resolved from registry ────────────────────────────
_TILES_JSON_DIR = ROOT / "tiles_locations_json"


def _model_tiles_json(model_name: str) -> Path:
    entry = _registry_entry(model_name)
    if entry and "dataset_name" in entry:
        p = _TILES_JSON_DIR / f"{entry['dataset_name']}.json"
        if p.exists():
            return p
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
    entry = _registry_entry(model_name)
    if entry and "dataset_name" in entry:
        p = ROOT / "data" / entry["dataset_name"]
        if p.exists():
            return p
    from src.config import DATA_DIR
    return DATA_DIR


@st.cache_data(show_spinner=False)
def tile_map(model_name: str) -> dict:
    from src.dataset import load_tiles_json
    return {t["imagery_file"]: t for t in load_tiles_json(_model_tiles_json(model_name))}


# ── PNG paths ─────────────────────────────────────────────────────────────────
def _png_path(imagery_file: str, style: str, subdir: str = "") -> Path:
    stem   = Path(imagery_file).stem
    suffix = "_overlay" if style == "overlay" else ""
    folder = TILES_DIR / (subdir if subdir else model_stem)
    return folder / f"{stem}{suffix}.png"


def _generate_pngs(imagery_file: str, metrics_row: dict) -> tuple[Path, Path]:
    """Generate side-by-side and overlay PNGs for a training tile."""
    from src.dataset import TileDataset
    from src.visualize import save_tile_overlay_png, save_tile_png

    p1 = _png_path(imagery_file, "masks")
    p2 = _png_path(imagery_file, "overlay")

    if p1.exists() and p2.exists():
        return p1, p2

    entry = tile_map(selected_model_name).get(imagery_file)
    if entry is None:
        return p1, p2

    model, device = get_model(str(selected_model_path))
    ds = TileDataset([entry], data_dir=_model_data_dir(selected_model_name))
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


# ── Test tile loading + inference ─────────────────────────────────────────────
_MODEL_BANDS = ["RED", "GREEN", "BLUE", "DSM_NORMALIZED"]
_TRAIN_MEAN  = np.array([150.73301134918557, 123.75755228360018,
                          92.57823716578613,  -9.734063808604613], dtype=np.float32)
_TRAIN_STD   = np.array([39.721974708734216,  34.06117915518031,
                          30.092062243775406,   4.684211737168346], dtype=np.float32)
_IGNORE_IDX  = 255


def _load_test_tile(imagery_file: str, mask_file: str):
    """
    Load a test tile from _TEST_DATA_DIR.
    Returns (img_chw float32 [4,H,W] normalised, mask_hw uint8 remapped).
    """
    img_npz  = np.load(_TEST_DATA_DIR / imagery_file)
    img_hw_c = np.stack([img_npz[b].astype(np.float32) for b in _MODEL_BANDS], axis=-1)
    img_norm = (img_hw_c - _TRAIN_MEAN) / _TRAIN_STD
    img_chw  = img_norm.transpose(2, 0, 1)  # (4, H, W)

    mask_npz = np.load(_TEST_DATA_DIR / mask_file)
    mask_raw = mask_npz[list(mask_npz.keys())[0]]
    if mask_raw.ndim == 3:
        mask_raw = mask_raw.squeeze(0)
    mask_raw = mask_raw.astype(np.uint8)

    mask = np.full_like(mask_raw, _IGNORE_IDX)
    mask[mask_raw == 1]  = 0
    mask[mask_raw == 14] = 1
    return img_chw, mask


def _generate_test_pngs(imagery_file: str, mask_file: str,
                         metrics_row: dict) -> tuple[Path, Path]:
    """Generate PNGs for a test tile. Cached in output/tiles/test_<model_stem>/."""
    from src.visualize import save_tile_overlay_png, save_tile_png

    subdir = f"test_{model_stem}"
    p1 = _png_path(imagery_file, "masks",   subdir)
    p2 = _png_path(imagery_file, "overlay", subdir)

    if p1.exists() and p2.exists():
        return p1, p2

    if not (_TEST_DATA_DIR / imagery_file).exists():
        return p1, p2

    img_chw, mask = _load_test_tile(imagery_file, mask_file)

    model, device = get_model(str(selected_model_path))
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


# ── Shared metric display helper ──────────────────────────────────────────────
def _show_tile_metrics(full: dict) -> None:
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


# ── Test DuckDB connection (module scope so both tab_explorer and tab_test can use it)
@st.cache_resource
def get_test_con(path: str) -> duckdb.DuckDBPyConnection:
    _c = duckdb.connect()
    _c.execute(f"CREATE VIEW test_metrics AS SELECT * FROM read_parquet('{path}')")
    return _c


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_explorer, tab_map, tab_compare, tab_test, tab_data = st.tabs(
    ["Overview", "Tile explorer", "Map", "Compare models", "Test set", "Raw data"]
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

    if _has_cm:
        _tp, _fp, _fn = float(kpi["sum_tp"]), float(kpi["sum_fp"]), float(kpi["sum_fn"])
        _denom = 2 * _tp + _fp + _fn
        global_f1 = (2 * _tp / _denom) if _denom > 0 else float("nan")
    else:
        global_f1 = None

    # ── Test KPIs (computed early so they can be shown alongside train KPIs) ──
    _ov_tmet_early = _test_metrics_path(model_stem)
    _ov_t_n_tiles = _ov_t_n_ero = None
    _ov_t_f1_mean = _ov_t_global_f1 = _ov_t_recall = _ov_t_f1_no = None
    if _ov_tmet_early:
        _ov_tc = duckdb.connect()
        _ov_t_has_cm = "tp_erosion" in _ov_tc.execute(
            f"SELECT * FROM read_parquet('{_ov_tmet_early}') LIMIT 0"
        ).df().columns
        _ov_t_cm = (
            ", SUM(tp_erosion) AS t_tp, SUM(fp_erosion) AS t_fp, SUM(fn_erosion) AS t_fn"
            if _ov_t_has_cm else ""
        )
        _ov_tkpi = _ov_tc.execute(f"""
            SELECT COUNT(*) AS n_tiles,
                   COUNTIF(n_erosion_pixels > 0) AS n_ero,
                   AVG(CASE WHEN n_erosion_pixels > 0 THEN f1_erosion     END) AS f1_mean,
                   AVG(CASE WHEN n_erosion_pixels > 0 THEN recall_erosion END) AS rec_mean,
                   AVG(f1_no_erosion) AS f1_no
                   {_ov_t_cm}
            FROM read_parquet('{_ov_tmet_early}')
        """).df().iloc[0]
        _ov_t_n_tiles = int(_ov_tkpi["n_tiles"])
        _ov_t_n_ero   = int(_ov_tkpi["n_ero"])
        _ov_t_f1_mean = float(_ov_tkpi["f1_mean"])
        _ov_t_recall  = float(_ov_tkpi["rec_mean"])
        _ov_t_f1_no   = float(_ov_tkpi["f1_no"])
        if _ov_t_has_cm:
            _tt, _tf, _tn = (float(_ov_tkpi["t_tp"]), float(_ov_tkpi["t_fp"]),
                             float(_ov_tkpi["t_fn"]))
            _td = 2 * _tt + _tf + _tn
            _ov_t_global_f1 = (2 * _tt / _td) if _td > 0 else float("nan")

    def _t_delta(val, label: str = "Test", fmt: str = ".4f") -> str | None:
        """Return delta string for st.metric, or None if no test data."""
        if val is None or (isinstance(val, float) and val != val):
            return None
        if fmt == ",d":
            return f"{label}: {int(val):,}"
        return f"{label}: {val:{fmt}}"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Tiles evaluated",      f"{int(kpi['n_tiles']):,}",
              delta=_t_delta(_ov_t_n_tiles, "Test", ",d"), delta_color="off")
    c2.metric("Tiles with erosion",   f"{int(kpi['n_tiles_with_erosion']):,}",
              help="Ground-truth has ≥1 erosion pixel",
              delta=_t_delta(_ov_t_n_ero, "Test", ",d"), delta_color="off")
    c3.metric("Mean F1 erosion",      f"{kpi['f1_erosion_nonzero']:.4f}",
              help="Macro: average of per-tile F1, on erosion tiles only",
              delta=_t_delta(_ov_t_f1_mean), delta_color="off")
    if global_f1 is not None:
        c4.metric("Global F1 erosion", f"{global_f1:.4f}",
                  help="Micro: 2·ΣTP / (2·ΣTP + ΣFP + ΣFN) across all pixels",
                  delta=_t_delta(_ov_t_global_f1), delta_color="off")
    else:
        c4.metric("Global F1 erosion", "—",
                  delta=_t_delta(_ov_t_global_f1), delta_color="off")
    c5.metric("Recall (erosion)",     f"{kpi['recall_nonzero']:.4f}",
              delta=_t_delta(_ov_t_recall), delta_color="off")
    c6.metric("F1 no-erosion (ref.)", f"{kpi['f1_no_erosion']:.4f}",
              delta=_t_delta(_ov_t_f1_no), delta_color="off")

    max_px_val = int(q("SELECT MAX(n_erosion_pixels) FROM metrics").iloc[0, 0])
    min_px = st.slider(
        "Min erosion pixels — raise to exclude small/sparse erosion patches",
        min_value=0, max_value=max_px_val, value=0, step=50,
        key="overview_min_px",
    )

    _eff_min = max(min_px, 1)
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

    # ── Filtered test KPIs ────────────────────────────────────────────────────
    _dov_t_n = _dov_t_f1 = _dov_t_gf1 = _dov_t_prec = _dov_t_rec = _dov_t_f1no = None
    if _ov_tmet_early:
        _dtc = duckdb.connect()
        _dt_has_cm = "tp_erosion" in _dtc.execute(
            f"SELECT * FROM read_parquet('{_ov_tmet_early}') LIMIT 0"
        ).df().columns
        _dt_cm = (
            ", SUM(tp_erosion) AS t_tp, SUM(fp_erosion) AS t_fp, SUM(fn_erosion) AS t_fn"
            if _dt_has_cm else ""
        )
        _dtkpi = _dtc.execute(f"""
            SELECT COUNT(*) AS n_tiles,
                   AVG(f1_erosion) AS f1,
                   AVG(precision_erosion) AS prec,
                   AVG(recall_erosion) AS rec,
                   AVG(f1_no_erosion) AS f1_no
                   {_dt_cm}
            FROM read_parquet('{_ov_tmet_early}')
            WHERE n_erosion_pixels >= {_eff_min}
        """).df().iloc[0]
        _dov_t_n    = int(_dtkpi["n_tiles"])
        _dov_t_f1   = float(_dtkpi["f1"])
        _dov_t_prec = float(_dtkpi["prec"])
        _dov_t_rec  = float(_dtkpi["rec"])
        _dov_t_f1no = float(_dtkpi["f1_no"])
        if _dt_has_cm:
            _dtt, _dtf, _dtn = (float(_dtkpi["t_tp"]), float(_dtkpi["t_fp"]),
                                float(_dtkpi["t_fn"]))
            _dtd = 2 * _dtt + _dtf + _dtn
            _dov_t_gf1 = (2 * _dtt / _dtd) if _dtd > 0 else float("nan")

    def _fmt(v: object, dec: int = 4) -> str:
        try:
            f = float(v)
            return "N/A" if (f != f) else f"{f:.{dec}f}"
        except (TypeError, ValueError):
            return "N/A"

    def _red_metric(label: str, value: str, test_value: str | None = None) -> str:
        _test_html = (
            f'<div style="color:#9B59B6;font-size:0.78em;margin-top:3px">Test: {test_value}</div>'
            if test_value else ""
        )
        return (
            f'<div style="background:#fff0f0;border-radius:8px;padding:10px 4px;'
            f'text-align:center;margin:2px">'
            f'<div style="color:#888;font-size:0.78em;margin-bottom:4px">{label}</div>'
            f'<div style="color:#c0392b;font-size:1.5em;font-weight:700">{value}</div>'
            f'<div style="color:#bbb;font-size:0.72em">≥ {_eff_min:,} px</div>'
            f'{_test_html}'
            f'</div>'
        )

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.markdown(_red_metric("Tiles (filtered)", f"{int(dkpi['n_tiles']):,}",
                            f"{_dov_t_n:,}" if _dov_t_n is not None else None),
                unsafe_allow_html=True)
    r2.markdown(_red_metric("Mean F1 erosion",  _fmt(dkpi['f1']),
                            _fmt(_dov_t_f1) if _dov_t_f1 is not None else None),
                unsafe_allow_html=True)
    r3.markdown(_red_metric("Global F1 erosion",
                            _fmt(d_global_f1) if d_global_f1 is not None else "—",
                            _fmt(_dov_t_gf1) if _dov_t_gf1 is not None else None),
                unsafe_allow_html=True)
    r4.markdown(_red_metric("Precision",        _fmt(dkpi['prec']),
                            _fmt(_dov_t_prec) if _dov_t_prec is not None else None),
                unsafe_allow_html=True)
    r5.markdown(_red_metric("Recall",           _fmt(dkpi['rec']),
                            _fmt(_dov_t_rec) if _dov_t_rec is not None else None),
                unsafe_allow_html=True)
    r6.markdown(_red_metric("F1 no-erosion",    _fmt(dkpi['f1_no']),
                            _fmt(_dov_t_f1no) if _dov_t_f1no is not None else None),
                unsafe_allow_html=True)

    st.divider()

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
        scatter, x="recall_erosion", y="precision_erosion",
        color="f1_erosion", size="_size",
        hover_data=["imagery_file", "f1_erosion", "n_erosion_pixels"],
        color_continuous_scale="RdYlGn",
        title=f"Each point = 1 tile · colour = F1 · size = erosion px · {len(scatter):,} tiles",
        labels={"recall_erosion": "Recall (erosion)", "precision_erosion": "Precision (erosion)"},
    )
    fig_scatter.update_layout(height=650)
    st.plotly_chart(fig_scatter, width='stretch')

    st.divider()
    st.subheader("Metric distributions")

    # Load test distribution for the same model (if available)
    _ov_tmet = _test_metrics_path(model_stem)
    dist_test = pd.DataFrame()
    if _ov_tmet:
        _test_con_ov = duckdb.connect()
        dist_test = _test_con_ov.execute(f"""
            SELECT f1_erosion, precision_erosion, recall_erosion, f1_no_erosion
            FROM read_parquet('{_ov_tmet}')
            WHERE n_erosion_pixels >= {min_px}
        """).df()

    # Selector — show "Both" only when test metrics exist
    _dist_options = ["Train", "Both (overlay)"] if not dist_test.empty else ["Train"]
    if not dist_test.empty:
        _dist_options = ["Train", "Test", "Both (overlay)"]
    dist_view = st.radio(
        "Show", _dist_options, index=0,
        horizontal=True, key="dist_view",
        help="'Both (overlay)' superimposes train and test to reveal distribution shift.",
    )

    dist = q(f"""
        SELECT f1_erosion, precision_erosion, recall_erosion, f1_no_erosion
        FROM metrics
        WHERE n_erosion_pixels >= {min_px}
    """)

    _TEST_PURPLE = "#9B59B6"

    def _hist(col: str, train_color: str, title: str) -> go.Figure:
        fig = go.Figure()
        show_train = dist_view in ("Train", "Both (overlay)")
        show_test  = dist_view in ("Test",  "Both (overlay)") and not dist_test.empty
        barmode    = "overlay" if (show_train and show_test) else "relative"

        if show_train:
            fig.add_trace(go.Histogram(
                x=dist[col], name="Train",
                marker_color=train_color,
                opacity=0.65 if show_test else 0.85,
                nbinsx=60,
            ))
        if show_test:
            fig.add_trace(go.Histogram(
                x=dist_test[col], name="Test",
                marker_color=_TEST_PURPLE,
                opacity=0.65 if show_train else 0.85,
                nbinsx=60,
            ))

        fig.update_layout(
            barmode=barmode,
            title=title,
            legend=dict(orientation="h", y=1.12, x=0),
            margin=dict(t=50),
            height=350,
        )
        fig.update_xaxes(range=[0, 1])
        return fig

    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_hist("f1_erosion",        "#e05252", "F1 erosion"),
                        width='stretch')
        st.plotly_chart(_hist("recall_erosion",    "#5288e0", "Recall erosion"),
                        width='stretch')
    with col_b:
        st.plotly_chart(_hist("precision_erosion", "#52c4a0", "Precision erosion"),
                        width='stretch')
        st.plotly_chart(_hist("f1_no_erosion",     "#c084e0", "F1 no-erosion (reference)"),
                        width='stretch')

    _n_train = len(dist)
    _n_test  = len(dist_test) if not dist_test.empty else 0
    if _n_test:
        st.caption(
            f"Train: **{_n_train:,}** tiles · Test: **{_n_test:,}** tiles "
            f"· filter: ≥ {min_px} erosion px"
        )
    else:
        st.caption(
            f"**{_n_train:,}** tiles · filter: ≥ {min_px} erosion px  "
            f"· _(run `python -m DASHBOARD_FROM_TYTONAI.evaluate_test` to unlock Test / Both views)_"
        )

    st.divider()
    st.info(
        "**How to read erosion metrics:**  \n"
        "- **Low recall erosion** → under-prediction: the model *misses* real erosion (false negatives)  \n"
        "- **Low precision erosion** → over-prediction: the model *hallucinates* erosion (false positives)  \n"
        "These two metrics describe opposite failure modes; F1 penalises both at once."
    )

    wc, bc = st.columns(2)
    with wc:
        st.subheader("Worst 10 — F1 erosion")
        st.dataframe(q("""
            SELECT imagery_file,
                   ROUND(f1_erosion, 4)        AS f1_ero,
                   ROUND(precision_erosion, 4) AS prec,
                   ROUND(recall_erosion, 4)    AS rec,
                   n_erosion_pixels             AS ero_px
            FROM metrics WHERE n_erosion_pixels > 0
            ORDER BY f1_erosion ASC LIMIT 10
        """), width="stretch", hide_index=True)
    with bc:
        st.subheader("Best 10 — F1 erosion")
        st.dataframe(q("""
            SELECT imagery_file,
                   ROUND(f1_erosion, 4)        AS f1_ero,
                   ROUND(precision_erosion, 4) AS prec,
                   ROUND(recall_erosion, 4)    AS rec,
                   n_erosion_pixels             AS ero_px
            FROM metrics ORDER BY f1_erosion DESC LIMIT 10
        """), width="stretch", hide_index=True)

    st.divider()
    oc, uc = st.columns(2)
    with oc:
        st.subheader("Worst 10 — Precision erosion  (over-prediction)")
        st.dataframe(q("""
            SELECT imagery_file,
                   ROUND(precision_erosion, 4) AS prec_ero,
                   ROUND(recall_erosion, 4)    AS rec_ero,
                   ROUND(f1_erosion, 4)        AS f1_ero,
                   n_erosion_pixels             AS ero_px
            FROM metrics WHERE n_erosion_pixels > 0
            ORDER BY precision_erosion ASC LIMIT 10
        """), width="stretch", hide_index=True)
    with uc:
        st.subheader("Worst 10 — Recall erosion  (under-prediction)")
        st.dataframe(q("""
            SELECT imagery_file,
                   ROUND(recall_erosion, 4)    AS rec_ero,
                   ROUND(precision_erosion, 4) AS prec_ero,
                   ROUND(f1_erosion, 4)        AS f1_ero,
                   n_erosion_pixels             AS ero_px
            FROM metrics WHERE n_erosion_pixels > 0
            ORDER BY recall_erosion ASC LIMIT 10
        """), width="stretch", hide_index=True)

    st.divider()
    st.subheader("Worst 10 — F1 no-erosion  (erosion hallucination)")
    st.dataframe(q("""
        SELECT imagery_file,
               ROUND(f1_no_erosion, 4)        AS f1_no_ero,
               ROUND(precision_no_erosion, 4) AS prec_no_ero,
               ROUND(recall_no_erosion, 4)    AS rec_no_ero,
               n_no_erosion_pixels             AS no_ero_px,
               n_erosion_pixels                AS ero_px,
               ROUND(f1_erosion, 4)            AS f1_ero
        FROM metrics
        WHERE n_no_erosion_pixels >= 10000
        ORDER BY f1_no_erosion ASC LIMIT 10
    """), width="stretch", hide_index=True)


# ── TAB 2: Tile explorer ──────────────────────────────────────────────────────
with tab_explorer:
    st.subheader("Tile explorer")

    # ── Test set section (shown first) ────────────────────────────────────────
    with st.expander("Test set tiles", expanded=True):
        _exp_tmet = _test_metrics_path(model_stem)
        if _exp_tmet is None:
            st.info(
                f"No test metrics for **{selected_model_name}**.  \n"
                "Run: `python -m DASHBOARD_FROM_TYTONAI.evaluate_test` then refresh."
            )
        else:
            _exp_tcon = get_test_con(str(_exp_tmet))

            def _etq(query: str) -> pd.DataFrame:
                return _exp_tcon.execute(query).df()

            _exp_t_has_cm = "tp_erosion" in _exp_tcon.execute(
                "SELECT * FROM test_metrics LIMIT 0"
            ).df().columns

            # Mini KPI strip
            _exp_kpi = _etq("""
                SELECT COUNT(*) AS n_tiles,
                       COUNTIF(n_erosion_pixels > 0) AS n_ero,
                       AVG(CASE WHEN n_erosion_pixels > 0 THEN f1_erosion END) AS f1_mean,
                       AVG(CASE WHEN n_erosion_pixels > 0 THEN recall_erosion END) AS rec_mean
                FROM test_metrics
            """).iloc[0]

            _ekc1, _ekc2, _ekc3, _ekc4 = st.columns(4)
            _ekc1.metric("Test tiles",      f"{int(_exp_kpi['n_tiles']):,}")
            _ekc2.metric("Erosion tiles",   f"{int(_exp_kpi['n_ero']):,}")
            _ekc3.metric("Mean F1 erosion", _fmt(float(_exp_kpi['f1_mean'])))
            _ekc4.metric("Recall erosion",  _fmt(float(_exp_kpi['rec_mean'])))

            # Filters
            _ef1, _ef2, _ef3 = st.columns([3, 2, 2])
            with _ef1:
                _exp_t_sort_opts = {
                    "★ Worst F1 erosion":             "f1_erosion ASC, n_erosion_pixels DESC",
                    "★ Best F1 erosion":              "f1_erosion DESC, n_erosion_pixels DESC",
                    "★ Worst recall (miss erosion)":  "recall_erosion ASC, n_erosion_pixels DESC",
                    "★ Worst precision (false alarm)":"precision_erosion ASC, n_erosion_pixels DESC",
                    "Most GT erosion px":             "n_erosion_pixels DESC",
                }
                _exp_t_sort = st.selectbox("Sort by", list(_exp_t_sort_opts.keys()),
                                           index=0, key="exp_t_sort")
                _exp_t_order = _exp_t_sort_opts[_exp_t_sort]
            with _ef2:
                _exp_t_f1_range = st.slider("F1 erosion range", 0.0, 1.0, (0.0, 1.0),
                                            step=0.01, key="exp_t_f1")
            with _ef3:
                _exp_t_max_px = int(_etq("SELECT MAX(n_erosion_pixels) FROM test_metrics").iloc[0, 0])
                _exp_t_min_px = st.number_input("Min GT erosion px", 0, _exp_t_max_px, 0,
                                                step=100, key="exp_t_min_px")

            _exp_t_pred_col = (
                "(tp_erosion + fp_erosion) AS pred_ero_px" if _exp_t_has_cm
                else "NULL AS pred_ero_px"
            )
            _exp_t_filtered = _etq(f"""
                SELECT imagery_file, mask_file,
                       ROUND(f1_erosion, 4)        AS f1_erosion,
                       ROUND(precision_erosion, 4) AS precision,
                       ROUND(recall_erosion, 4)    AS recall,
                       n_erosion_pixels             AS gt_ero_px,
                       n_no_erosion_pixels          AS gt_no_ero_px,
                       {_exp_t_pred_col}
                FROM test_metrics
                WHERE f1_erosion BETWEEN {_exp_t_f1_range[0]} AND {_exp_t_f1_range[1]}
                  AND n_erosion_pixels >= {_exp_t_min_px}
                ORDER BY {_exp_t_order}
            """)

            _exp_t_total = len(_exp_t_filtered)
            _exp_t_pg_size = 20
            _exp_t_pages   = max(1, (_exp_t_total - 1) // _exp_t_pg_size + 1)
            _ep1, _ep2 = st.columns([4, 1])
            with _ep1:
                st.caption(f"**{_exp_t_total:,}** test tiles match filters")
            with _ep2:
                _exp_t_page = st.number_input("Page", 1, _exp_t_pages, 1, key="exp_t_page")

            _exp_t_page_df = _exp_t_filtered.iloc[
                (_exp_t_page - 1) * _exp_t_pg_size : _exp_t_page * _exp_t_pg_size
            ]
            _exp_t_event = st.dataframe(
                _exp_t_page_df, width='stretch', hide_index=True,
                on_select="rerun", selection_mode="single-row", key="exp_test_table",
            )

            _exp_t_sel = _exp_t_event.selection.rows if _exp_t_event.selection else []
            if not _exp_t_sel:
                st.caption("Click a row to visualise the tile.")
            else:
                _exp_t_img  = _exp_t_page_df.iloc[_exp_t_sel[0]]["imagery_file"]
                _exp_t_mask = _exp_t_page_df.iloc[_exp_t_sel[0]]["mask_file"]
                st.subheader("Selected test tile")
                _etc1, _etc2 = st.columns(2)
                with _etc1:
                    st.markdown("**Imagery file**")
                    st.code(_exp_t_img, language=None)
                with _etc2:
                    st.markdown("**Mask file**")
                    st.code(_exp_t_mask, language=None)

                _exp_t_full = _etq(
                    f"SELECT * FROM test_metrics WHERE imagery_file = '{_exp_t_img}' LIMIT 1"
                ).iloc[0].to_dict()

                with st.spinner("Generating test tile visualisation…"):
                    try:
                        _etp1, _etp2 = _generate_test_pngs(_exp_t_img, _exp_t_mask, _exp_t_full)
                    except Exception as _e:
                        st.error(f"Generation failed: {_e}")
                        _etp1 = _png_path(_exp_t_img, "masks",   f"test_{model_stem}")
                        _etp2 = _png_path(_exp_t_img, "overlay", f"test_{model_stem}")

                _exp_t_visu = st.radio(
                    "Visualisation", ["Side-by-side masks", "Contour overlay (RGB / DSM)"],
                    horizontal=True, key="exp_t_visu_mode",
                )
                _, _et_img_col, _ = st.columns([2, 6, 2])
                with _et_img_col:
                    _ep = _etp1 if _exp_t_visu == "Side-by-side masks" else _etp2
                    if _ep.exists():
                        st.image(str(_ep), width="stretch")
                    else:
                        st.warning("NPZ missing — run `evaluate_test` to download.")
                _show_tile_metrics(_exp_t_full)

    st.markdown("#### Training set tiles")

    sort_opts = {
        "★ Worst F1 + most erosion px":                  "f1_erosion ASC, n_erosion_pixels DESC",
        "★ Best F1 + most erosion px":                   "f1_erosion DESC, n_erosion_pixels DESC",
        "★ Worst precision erosion (over-prediction)":   "precision_erosion ASC, n_erosion_pixels DESC",
        "★ Worst recall erosion (under-prediction)":     "recall_erosion ASC, n_erosion_pixels DESC",
        "★ Worst F1 no-erosion (hallucination)":         "f1_no_erosion ASC, n_no_erosion_pixels DESC",
        "F1 erosion ↑ (worst first)":                    "f1_erosion ASC",
        "F1 erosion ↓ (best first)":                     "f1_erosion DESC",
        "Precision erosion ↑ (worst first)":             "precision_erosion ASC",
        "Recall erosion ↑ (worst first)":                "recall_erosion ASC",
        "F1 no-erosion ↑ (worst first)":                 "f1_no_erosion ASC",
        "F1 no-erosion ↓ (best first)":                  "f1_no_erosion DESC",
        "Most GT erosion px":                            "n_erosion_pixels DESC",
        "Least GT erosion px":                           "n_erosion_pixels ASC",
        "Most predicted erosion px (tp+fp)":             "tp_erosion + fp_erosion DESC",
        "Least predicted erosion px (tp+fp)":            "tp_erosion + fp_erosion ASC",
        "Most GT no-erosion px":                         "n_no_erosion_pixels DESC",
        "Largest tile (total px)":                       "n_erosion_pixels + n_no_erosion_pixels DESC",
        "Smallest tile (total px)":                      "n_erosion_pixels + n_no_erosion_pixels ASC",
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

    _max_total_px = int(q("SELECT MAX(n_erosion_pixels + n_no_erosion_pixels) FROM metrics").iloc[0, 0])
    _max_pred_px  = int(q("SELECT MAX(tp_erosion + fp_erosion) FROM metrics").iloc[0, 0])

    fc1, fc2, fc3, fc4, fc5, fc6 = st.columns([2, 2, 2, 2, 2, 1])
    with fc1:
        sort_label   = st.selectbox("Sort by (primary)", list(sort_opts.keys()), index=0)
        order_clause = sort_opts[sort_label]
    with fc2:
        sec_label     = st.selectbox("Then by (secondary)", list(sec_opts.keys()), index=0)
        order_clause += sec_opts[sec_label]
    with fc3:
        f1_range = st.slider("F1 erosion range", 0.0, 1.0, (0.0, 1.0), step=0.01)
    with fc4:
        max_px      = int(q("SELECT MAX(n_erosion_pixels) FROM metrics").iloc[0, 0])
        exp_min_px  = st.number_input("Min GT erosion px", 0, max_px, 0, step=100)
    with fc5:
        min_total_px = st.number_input(
            "Min tile size (labeled px)", 0, _max_total_px, 50_000, step=10_000,
        )
        st.caption("full tile = 384×384 = 147 456 px")
    with fc6:
        page_size = st.selectbox("Rows / page", [20, 50, 100], index=0)

    rf1, rf2, _ = st.columns([3, 3, 6])
    with rf1:
        min_pred_px = st.number_input("Min predicted erosion px  (tp + fp)", 0, _max_pred_px, 0, step=500)
    with rf2:
        max_gt_px   = st.number_input("Max GT erosion px", 0, max_px, max_px, step=500)

    filtered = q(f"""
        SELECT imagery_file, mask_file,
               ROUND(f1_erosion, 4)        AS f1_erosion,
               ROUND(precision_erosion, 4) AS precision,
               ROUND(recall_erosion, 4)    AS recall,
               n_erosion_pixels            AS gt_ero_px,
               n_no_erosion_pixels         AS gt_no_ero_px,
               (tp_erosion + fp_erosion)   AS pred_ero_px,
               (n_erosion_pixels + n_no_erosion_pixels) AS total_px
        FROM metrics
        WHERE f1_erosion BETWEEN {f1_range[0]} AND {f1_range[1]}
          AND n_erosion_pixels >= {exp_min_px}
          AND n_erosion_pixels <= {max_gt_px}
          AND (tp_erosion + fp_erosion) >= {min_pred_px}
          AND (n_erosion_pixels + n_no_erosion_pixels) >= {min_total_px}
        ORDER BY {order_clause}
    """)

    search_str = st.text_input("Search tile name (substring)", value="", placeholder="e.g. 81bdc548")
    if search_str:
        filtered = filtered[filtered["imagery_file"].str.contains(search_str, case=False, na=False)]

    total       = len(filtered)
    total_pages = max(1, (total - 1) // page_size + 1)
    pc1, pc2    = st.columns([4, 1])
    with pc1:
        st.caption(f"**{total:,}** tiles match filters")
    with pc2:
        page_num = st.number_input("Page", 1, total_pages, 1, key="exp_page")

    page_df  = filtered.iloc[(page_num - 1) * page_size : page_num * page_size]
    event    = st.dataframe(
        page_df, width='stretch', hide_index=True,
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

        nc1, nc2, nc3 = st.columns([3, 3, 2])
        with nc1:
            st.markdown("**Imagery file**")
            st.code(imagery_file, language=None)
        with nc2:
            st.markdown("**Mask file**")
            st.code(mask_file, language=None)
        with nc3:
            from src.build_geo import geo_parquet_path as _geo_path_exp
            _geo_f = _geo_path_exp(_registry_entry(selected_model_name).get("dataset_name", "default"))
            if _geo_f.exists():
                _geo_df  = pd.read_parquet(_geo_f, columns=["imagery_file", "lat", "lon"])
                _geo_row = _geo_df[_geo_df["imagery_file"] == imagery_file]
                if not _geo_row.empty:
                    st.markdown("**Location (centroid)**")
                    st.metric("Lat", f"{float(_geo_row.iloc[0]['lat']):.5f}")
                    st.metric("Lon", f"{float(_geo_row.iloc[0]['lon']):.5f}")

        full = q(f"SELECT * FROM metrics WHERE imagery_file = '{imagery_file}' LIMIT 1").iloc[0].to_dict()

        with st.spinner("Generating visualisation (cached after first run)…"):
            try:
                p1, p2 = _generate_pngs(imagery_file, full)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                p1, p2 = _png_path(imagery_file, "masks"), _png_path(imagery_file, "overlay")

        visu_mode = st.radio(
            "Visualisation", ["Side-by-side masks", "Contour overlay (RGB / DSM)"],
            horizontal=True, key="visu_mode",
        )
        _, img_col, _ = st.columns([2, 6, 2])
        with img_col:
            p = p1 if visu_mode == "Side-by-side masks" else p2
            if p.exists():
                st.image(str(p), width="stretch")
            else:
                st.warning("NPZ missing — cannot generate")

        _show_tile_metrics(full)


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
            f"```\npython -m src.run_all\n```\nto build it automatically."
        )
    else:
        @st.cache_data(show_spinner="Loading geo index…")
        def load_map_data(geo_path: str, metrics_path: str) -> pd.DataFrame:
            _con = duckdb.connect()
            return _con.execute(f"""
                SELECT g.imagery_file, g.lat, g.lon,
                       g.pixel_size_m, g.width_px, g.height_px,
                       m.f1_erosion, m.precision_erosion, m.recall_erosion,
                       m.iou_erosion, m.f1_no_erosion,
                       m.n_erosion_pixels, m.n_no_erosion_pixels, m.mask_file
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
            # ── Controls row ──────────────────────────────────────────────────
            map_ctrl1, map_ctrl2 = st.columns([6, 2])
            with map_ctrl1:
                jump_str = st.text_input(
                    "Jump to tile (paste name or UUID fragment from Explorer)",
                    value="", placeholder="e.g. 16413f59", key="map_jump",
                )
            with map_ctrl2:
                # Test tile toggle — only show if test geo + metrics exist
                _tgeo    = _test_geo_path()
                _tmet    = _test_metrics_path(model_stem)
                _can_test = _tgeo.exists() and _tmet is not None
                show_test_map = st.checkbox(
                    "Show test set tiles (purple)",
                    value=_can_test,
                    disabled=not _can_test,
                    key="map_show_test",
                    help="Run `python -m DASHBOARD_FROM_TYTONAI.evaluate_test` to enable."
                    if not _can_test else "",
                )

            _jump_row = None
            if jump_str:
                _hits = map_df_full[map_df_full["imagery_file"].str.contains(jump_str, case=False, na=False)]
                if not _hits.empty:
                    _jump_row = _hits.iloc[0]
                    st.success(f"Found: **{_jump_row['imagery_file']}** — lat {_jump_row['lat']:.5f}, lon {_jump_row['lon']:.5f}")
                else:
                    st.warning("No tile found matching that name.")

            mc1, mc2, mc3 = st.columns([2, 2, 2])
            with mc1:
                map_min_px = st.slider("Min erosion pixels", 0,
                    int(map_df_full["n_erosion_pixels"].max()), 0, step=50, key="map_min_px")
            with mc2:
                map_f1_range = st.slider("F1 erosion range", 0.0, 1.0, (0.0, 1.0),
                    step=0.01, key="map_f1_range")
            with mc3:
                color_metric = st.selectbox("Colour by",
                    ["recall_erosion", "f1_erosion", "precision_erosion",
                     "n_erosion_pixels", "pixel_size_m"],
                    index=0, key="map_color")

            map_df = map_df_full[
                (map_df_full["n_erosion_pixels"] >= map_min_px)
                & (map_df_full["f1_erosion"].between(*map_f1_range))
            ].copy()

            map_df["_highlight"] = (
                map_df["imagery_file"].str.contains(jump_str, case=False, na=False)
                if jump_str else False
            )

            # ── Load test tile data for overlay ───────────────────────────────
            test_map_df = pd.DataFrame()
            if show_test_map and _can_test:
                @st.cache_data(show_spinner="Loading test geo…")
                def _load_test_map(geo_path: str, metrics_path: str) -> pd.DataFrame:
                    _c = duckdb.connect()
                    return _c.execute(f"""
                        SELECT g.imagery_file, g.lat, g.lon,
                               g.capture_name, g.pixel_size_m,
                               m.f1_erosion, m.precision_erosion, m.recall_erosion,
                               m.n_erosion_pixels, m.mask_file
                        FROM read_parquet('{geo_path}') g
                        JOIN read_parquet('{metrics_path}') m
                          ON g.imagery_file = m.imagery_file
                    """).df()

                test_map_df = _load_test_map(str(_tgeo), str(_tmet))
                st.caption(
                    f"**{len(map_df):,}** train tiles  ·  "
                    f"**{len(test_map_df):,}** test tiles (purple)"
                )
            else:
                st.caption(f"**{len(map_df):,}** tiles shown")

            if map_df.empty:
                st.info("No tiles match the current filters.")
            else:
                # Centre on combined bbox when test tiles are shown
                if not test_map_df.empty:
                    all_lats = pd.concat([map_df["lat"], test_map_df["lat"]])
                    all_lons = pd.concat([map_df["lon"], test_map_df["lon"]])
                    _center_lat = all_lats.mean()
                    _center_lon = all_lons.mean()
                    _zoom = 10
                elif _jump_row is not None:
                    _center_lat = float(_jump_row["lat"])
                    _center_lon = float(_jump_row["lon"])
                    _zoom = 15
                else:
                    _center_lat = map_df["lat"].mean()
                    _center_lon = map_df["lon"].mean()
                    _zoom = 11

                cap = map_df["n_erosion_pixels"].quantile(0.95) + 1
                map_df["_size"] = map_df["n_erosion_pixels"].clip(upper=cap) + 1
                if _jump_row is not None:
                    map_df.loc[map_df["_highlight"], "_size"] = map_df["_size"].max() * 3

                fig_map = px.scatter_map(
                    map_df,
                    lat="lat", lon="lon",
                    color=color_metric,
                    size="_size",
                    hover_name="imagery_file",
                    hover_data={
                        "f1_erosion": ":.3f", "precision_erosion": ":.3f",
                        "recall_erosion": ":.3f", "n_erosion_pixels": True,
                        "_size": False, "lat": False, "lon": False,
                    },
                    color_continuous_scale="RdYlGn",
                    center={"lat": _center_lat, "lon": _center_lon},
                    zoom=_zoom, height=650,
                    map_style="carto-darkmatter",
                    title=f"Tiles coloured by {color_metric} · {len(map_df):,} tiles",
                )
                fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
                # Make train-tile markers more opaque
                fig_map.data[0].update(marker=dict(opacity=0.95))

                # ── Jump highlight overlay ─────────────────────────────────────
                if _jump_row is not None:
                    fig_map.add_trace(go.Scattermap(
                        lat=[float(_jump_row["lat"])], lon=[float(_jump_row["lon"])],
                        mode="markers",
                        marker=dict(size=24, color="#00FFFF", opacity=1.0),
                        name="", hovertext=str(_jump_row["imagery_file"]),
                        hoverinfo="text", showlegend=False,
                    ))
                    fig_map.add_trace(go.Scattermap(
                        lat=[float(_jump_row["lat"])], lon=[float(_jump_row["lon"])],
                        mode="markers",
                        marker=dict(size=40, color="rgba(0,255,255,0.25)"),
                        name="", hoverinfo="skip", showlegend=False,
                    ))

                # ── Purple test tile overlay ───────────────────────────────────
                _test_trace_idx = None
                if not test_map_df.empty:
                    _test_trace_idx = len(fig_map.data)   # will be next trace index
                    # Prefix hover names with "TEST:" so the click handler can tell them apart
                    _test_hover = "TEST:" + test_map_df["imagery_file"]
                    fig_map.add_trace(go.Scattermap(
                        lat=test_map_df["lat"],
                        lon=test_map_df["lon"],
                        mode="markers",
                        marker=dict(size=8, color="#9B59B6", opacity=0.95),
                        name="Test set",
                        hovertext=_test_hover,
                        hoverinfo="text",
                        customdata=test_map_df[
                            ["imagery_file", "mask_file", "f1_erosion",
                             "precision_erosion", "recall_erosion",
                             "n_erosion_pixels", "capture_name"]
                        ].values,
                        showlegend=True,
                    ))

                map_event = st.plotly_chart(
                    fig_map, width='stretch',
                    on_select="rerun", key="map_chart",
                )

                # ── Click handler ─────────────────────────────────────────────
                pts = (map_event.selection.points
                       if map_event and map_event.selection else [])

                if pts:
                    pt = pts[0]
                    htext = pt.get("hovertext") or ""
                    is_test = htext.startswith("TEST:")

                    if is_test:
                        # Retrieve info from customdata
                        cd = pt.get("customdata", [])
                        clicked_file = str(cd[0]) if len(cd) > 0 else htext[5:]
                        clicked_mask = str(cd[1]) if len(cd) > 1 else ""
                        st.divider()
                        st.subheader(f"Test tile: `{Path(clicked_file).stem}`")
                        st.caption(f"Capture: {str(cd[6]) if len(cd) > 6 else ''}")

                        nc1, nc2 = st.columns(2)
                        with nc1:
                            st.markdown("**Imagery file**")
                            st.code(clicked_file, language=None)
                        with nc2:
                            st.markdown("**Mask file**")
                            st.code(clicked_mask, language=None)

                        full_test = pd.read_parquet(_tmet)[
                            lambda df: df["imagery_file"] == clicked_file
                        ]
                        if not full_test.empty:
                            ft = full_test.iloc[0].to_dict()
                            with st.spinner("Generating test tile visualisation…"):
                                try:
                                    tp1, tp2 = _generate_test_pngs(clicked_file, clicked_mask, ft)
                                except Exception as e:
                                    st.error(f"Generation failed: {e}")
                                    tp1 = _png_path(clicked_file, "masks", f"test_{model_stem}")
                                    tp2 = _png_path(clicked_file, "overlay", f"test_{model_stem}")

                            vmode = st.radio(
                                "Visualisation",
                                ["Side-by-side masks", "Contour overlay (RGB / DSM)"],
                                horizontal=True, key="map_test_vmode",
                            )
                            _, ic, _ = st.columns([2, 6, 2])
                            with ic:
                                p = tp1 if vmode == "Side-by-side masks" else tp2
                                if p.exists():
                                    st.image(str(p), width="stretch")
                                else:
                                    st.warning("NPZ missing from test_data/ — run download first.")
                            _show_tile_metrics(ft)

                    else:
                        # Train tile
                        clicked_file = htext or pt.get("customdata", [None])[0]
                        if clicked_file:
                            st.divider()
                            st.subheader(f"Train tile: `{Path(clicked_file).stem}`")
                            nc1, nc2 = st.columns(2)
                            full_map = q(f"""
                                SELECT * FROM metrics WHERE imagery_file = '{clicked_file}' LIMIT 1
                            """).iloc[0].to_dict()
                            with nc1:
                                st.markdown("**Imagery file**")
                                st.code(clicked_file, language=None)
                            with nc2:
                                st.markdown("**Mask file**")
                                st.code(full_map.get("mask_file", ""), language=None)

                            with st.spinner("Generating visualisation…"):
                                try:
                                    p1, p2 = _generate_pngs(clicked_file, full_map)
                                except Exception as e:
                                    st.error(f"Generation failed: {e}")
                                    p1 = _png_path(clicked_file, "masks")
                                    p2 = _png_path(clicked_file, "overlay")

                            vmode_map = st.radio(
                                "Visualisation",
                                ["Side-by-side masks", "Contour overlay (RGB / DSM)"],
                                horizontal=True, key="map_visu_mode",
                            )
                            _, img_col_map, _ = st.columns([2, 6, 2])
                            with img_col_map:
                                p = p1 if vmode_map == "Side-by-side masks" else p2
                                if p.exists():
                                    st.image(str(p), width="stretch")
                                else:
                                    st.warning("NPZ missing — cannot generate")
                            _show_tile_metrics(full_map)


# ── TAB 4: Compare models ─────────────────────────────────────────────────────
with tab_compare:
    st.subheader("Model comparison")

    if not _REGISTRY_PATH.exists():
        st.warning("No `models_registry.json` found at repo root.")
    else:
        @st.cache_data(show_spinner="Loading aggregate metrics…")
        def _load_all_metrics(registry_hash: str) -> pd.DataFrame:
            rows = []
            for entry in list(_REGISTRY.values()):
                stem = Path(entry["model_file"]).stem
                pq = next(
                    (p for p in [OUTPUT_DIR / f"metrics_{stem}.parquet",
                                 OUTPUT_DIR / "metrics.parquet"] if p.exists()),
                    None,
                )
                if pq is None:
                    continue

                _c = duckdb.connect()
                _has_cm = "tp_erosion" in _c.execute(
                    f"SELECT * FROM read_parquet('{pq}') LIMIT 0"
                ).df().columns
                _cm_expr = (
                    ", SUM(tp_erosion) AS sum_tp"
                    ", SUM(fp_erosion) AS sum_fp"
                    ", SUM(fn_erosion) AS sum_fn"
                    if _has_cm else ""
                )
                agg = _c.execute(f"""
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
                total_px  = float(agg["total_erosion_px"]) + float(agg["total_no_erosion_px"])

                # ── Test set metrics (if available) ───────────────────────────
                tpq = OUTPUT_DIR / f"test_metrics_{stem}.parquet"
                test_gf1 = test_mf1 = test_prec = test_rec = test_iou = float("nan")
                if tpq.exists():
                    _tc = duckdb.connect()
                    _t_has_cm = "tp_erosion" in _tc.execute(
                        f"SELECT * FROM read_parquet('{tpq}') LIMIT 0"
                    ).df().columns
                    _t_cm = (
                        ", SUM(tp_erosion) AS t_tp, SUM(fp_erosion) AS t_fp, SUM(fn_erosion) AS t_fn"
                        if _t_has_cm else ""
                    )
                    tagg = _tc.execute(f"""
                        SELECT
                            AVG(CASE WHEN n_erosion_pixels > 0 THEN f1_erosion        END) AS t_f1,
                            AVG(CASE WHEN n_erosion_pixels > 0 THEN precision_erosion END) AS t_prec,
                            AVG(CASE WHEN n_erosion_pixels > 0 THEN recall_erosion    END) AS t_rec,
                            AVG(iou_erosion)                                               AS t_iou
                            {_t_cm}
                        FROM read_parquet('{tpq}')
                    """).df().iloc[0]

                    _tt = float(tagg.get("t_tp", float("nan")))
                    _tf = float(tagg.get("t_fp", float("nan")))
                    _tn = float(tagg.get("t_fn", float("nan")))
                    _td = 2 * _tt + _tf + _tn
                    test_gf1  = (2 * _tt / _td) if _td > 0 else float("nan")
                    test_mf1  = float(tagg["t_f1"])
                    test_prec = float(tagg["t_prec"])
                    test_rec  = float(tagg["t_rec"])
                    test_iou  = float(tagg["t_iou"])

                rows.append({
                    "model":              entry["model_file"],
                    "label":              stem,
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
                    # test set (NaN if not evaluated)
                    "test_global_f1":     round(test_gf1, 4)  if not np.isnan(test_gf1)  else float("nan"),
                    "test_mean_f1":       round(test_mf1, 4)  if not np.isnan(test_mf1)  else float("nan"),
                    "test_precision":     round(test_prec, 4) if not np.isnan(test_prec) else float("nan"),
                    "test_recall":        round(test_rec, 4)  if not np.isnan(test_rec)  else float("nan"),
                    "test_iou":           round(test_iou, 4)  if not np.isnan(test_iou)  else float("nan"),
                })
            return pd.DataFrame(rows)

        @st.cache_data(show_spinner="Loading tile-level data…")
        def _load_tile_data(registry_hash: str) -> dict[str, pd.DataFrame]:
            out = {}
            for entry in list(_REGISTRY.values()):
                stem = Path(entry["model_file"]).stem
                pq = next(
                    (p for p in [OUTPUT_DIR / f"metrics_{stem}.parquet",
                                 OUTPUT_DIR / "metrics.parquet"] if p.exists()),
                    None,
                )
                if pq is None:
                    continue
                out[entry["model_file"]] = pd.read_parquet(pq)
            return out

        @st.cache_data(show_spinner="Loading test tile-level data…")
        def _load_test_tile_data(registry_hash: str) -> dict[str, pd.DataFrame]:
            out = {}
            for entry in list(_REGISTRY.values()):
                stem = Path(entry["model_file"]).stem
                tpq  = OUTPUT_DIR / f"test_metrics_{stem}.parquet"
                if not tpq.exists():
                    continue
                out[entry["model_file"]] = pd.read_parquet(tpq)
            return out

        _reg_hash      = str(hash(_REGISTRY_PATH.read_text()))
        compare_df     = _load_all_metrics(_reg_hash)
        tile_data      = _load_tile_data(_reg_hash)
        test_tile_data = _load_test_tile_data(_reg_hash)

        if compare_df.empty:
            st.info("No metrics computed yet. Run `python -m src.run_all` to evaluate.")
        else:
            n_evaluated = len(compare_df)
            n_total     = len(_REGISTRY)
            n_test      = compare_df["test_global_f1"].notna().sum()
            st.caption(
                f"**{n_evaluated} / {n_total}** models evaluated "
                f"({n_test} with test-set metrics)"
            )

            # v2 was trained on the test set — show its results but exclude from winner/highlight
            _EXCLUDED = {"v2"}   # registry "version" values disqualified from competition

            _eligible = compare_df[~compare_df["version"].isin(_EXCLUDED)]

            ranked = _eligible.sort_values("global_f1_erosion", ascending=False).reset_index(drop=True)
            if ranked.empty:
                ranked = compare_df.sort_values("global_f1_erosion", ascending=False).reset_index(drop=True)
            winner = ranked.iloc[0]
            runner = ranked.iloc[1] if len(ranked) > 1 else None
            _gap   = (f"  (+{winner['global_f1_erosion'] - runner['global_f1_erosion']:.4f} vs runner-up)"
                      if runner is not None else "")
            st.success(
                f"🏆 **Winner (train): `{winner['label']}`** — "
                f"Global F1: **{winner['global_f1_erosion']:.4f}**{_gap}"
            )

            # Show test winner if available (also exclude v2)
            _test_ranked = _eligible.dropna(subset=["test_global_f1"]).sort_values(
                "test_global_f1", ascending=False
            )
            if not _test_ranked.empty:
                tw = _test_ranked.iloc[0]
                st.info(
                    f"**Test set winner: `{tw['label']}`** — "
                    f"Test Global F1: **{tw['test_global_f1']:.4f}**"
                )

            if compare_df["version"].isin(_EXCLUDED).any():
                st.warning(
                    "**v2** (`model_v2_no_erosion_td`) was trained on the test set — "
                    "its scores are shown for reference only and excluded from winner selection."
                )

            # ── Section 1: Leaderboard ─────────────────────────────────────────
            st.markdown("#### 1 · Leaderboard")
            st.caption(
                "⚠️ **Mean F1 macro ≠ F1(mean Precision, mean Recall)** — each metric is averaged "
                "independently per tile, so 2·P·R/(P+R) does not reconstruct Mean F1. "
                "Likewise, a high **Global F1 micro** (pixel-level) does not imply strong macro averages: "
                "it is dominated by large dense-erosion regions and can mask poor performance on sparse tiles."
            )

            _train_col_map = {
                "Mean F1 erosion (macro)":   "mean_f1_erosion",
                "Global F1 erosion (micro)": "global_f1_erosion",
                "Mean Recall":               "mean_recall",
                "Mean Precision":            "mean_precision",
                "Mean IOU erosion":          "mean_iou_erosion",
                "Mean F1 no-erosion":        "mean_f1_no_erosion",
                "Test Global F1":            "test_global_f1",
                "Test Mean F1":              "test_mean_f1",
                "Test Recall":               "test_recall",
                "Test Precision":            "test_precision",
            }
            _lb_sort = st.selectbox("Sort by", list(_train_col_map.keys()),
                                    index=1, key="lb_sort")
            _lb_col  = _train_col_map[_lb_sort]

            _lb_display_cols = [
                "label", "dataset", "epoch",
                "n_tiles", "n_erosion_tiles", "erosion_prevalence",
                "global_f1_erosion", "mean_f1_erosion",
                "mean_precision", "mean_recall",
                "mean_iou_erosion",
                "test_global_f1", "test_mean_f1",
                "test_precision", "test_recall", "test_iou",
            ]

            _train_num_cols = ["global_f1_erosion", "mean_f1_erosion",
                               "mean_precision", "mean_recall", "mean_iou_erosion"]
            _test_num_cols  = ["test_global_f1", "test_mean_f1",
                               "test_precision", "test_recall", "test_iou"]

            _lb_df = compare_df[_lb_display_cols].sort_values(
                _lb_col, ascending=False, na_position="last"
            ).reset_index(drop=True)

            # Rows whose version is excluded get a strikethrough-style dim instead of highlight
            _is_excluded = compare_df.set_index("label").reindex(
                _lb_df["label"]
            )["version"].isin(_EXCLUDED).values

            # Grey colour map for excluded models — used in all charts
            _v2_stems = set(compare_df[compare_df["version"].isin(_EXCLUDED)]["label"])
            _GREY = "#888888"
            # covers plain stem + " [Train]" / " [Test]" / " [Both]" suffixed variants
            _grey_cmap: dict[str, str] = {}
            for _s in _v2_stems:
                _grey_cmap[_s] = _GREY
                for _sfx in (" [Train]", " [Test]"):
                    _grey_cmap[_s + _sfx] = _GREY

            def _bar_grey(df: pd.DataFrame, col: str, title: str,
                          colorscale: str = "RdYlGn") -> go.Figure:
                """Horizontal bar chart: excluded models grey, others colored by value."""
                _active = df[~df["label"].isin(_v2_stems)].copy()
                _excl   = df[df["label"].isin(_v2_stems)].copy()
                fig = go.Figure()
                if not _active.empty:
                    _vals = _active[col].fillna(0).tolist()
                    fig.add_trace(go.Bar(
                        x=_vals, y=_active["label"].tolist(),
                        orientation="h",
                        marker=dict(color=_vals, colorscale=colorscale,
                                    cmin=0, cmax=1, showscale=False),
                        text=[f"{v:.4f}" if not pd.isna(v) else "—"
                              for v in _active[col]],
                        textposition="outside",
                        showlegend=False,
                    ))
                if not _excl.empty:
                    _vals_e = _excl[col].fillna(0).tolist()
                    fig.add_trace(go.Bar(
                        x=_vals_e, y=_excl["label"].tolist(),
                        orientation="h",
                        marker_color=_GREY,
                        text=[f"{v:.4f}" if not pd.isna(v) else "—"
                              for v in _excl[col]],
                        textposition="outside",
                        showlegend=False,
                    ))
                fig.update_layout(
                    title=title, barmode="overlay",
                    height=80 + 60 * len(df),
                    showlegend=False,
                    xaxis_title="Global F1 (micro)",
                    yaxis_title="",
                    margin=dict(l=0, r=60),
                )
                return fig

            def _hi_max(s: pd.Series) -> list[str]:
                """Highlight max only among eligible rows."""
                eligible_vals = s[~_is_excluded]
                best = eligible_vals.max() if not eligible_vals.empty else None
                return [
                    ("background-color:#555;color:#aaa" if _is_excluded[i] else
                     ("background-color:#1a472a;color:white" if (best is not None and v == best) else ""))
                    for i, v in enumerate(s)
                ]

            def _hi_purple_max(s: pd.Series) -> list[str]:
                eligible_vals = s[~_is_excluded].dropna()
                best = eligible_vals.max() if not eligible_vals.empty else None
                return [
                    ("background-color:#555;color:#aaa" if _is_excluded[i] else
                     ("background-color:#4a235a;color:white" if (best is not None and pd.notna(v) and v == best) else
                      ("background-color:#1a0a2a;color:#888" if pd.isna(v) else "background-color:#1a0a2a;color:#ccc")))
                    for i, v in enumerate(s)
                ]

            _col_labels = {
                "global_f1_erosion": "Global F1 (micro·px)",
                "mean_f1_erosion":   "Mean F1 (macro·ero tiles)",
                "mean_precision":    "Precision (macro·ero tiles)",
                "mean_recall":       "Recall (macro·ero tiles)",
                "mean_iou_erosion":  "IoU (macro·all tiles)",
                "test_global_f1":    "Test Global F1 (micro·px)",
                "test_mean_f1":      "Test Mean F1 (macro·ero tiles)",
                "test_precision":    "Test Precision (macro·ero tiles)",
                "test_recall":       "Test Recall (macro·ero tiles)",
                "test_iou":          "Test IoU (macro·all tiles)",
            }
            _lb_df_display = _lb_df.rename(columns=_col_labels)
            _train_num_display = [_col_labels.get(c, c) for c in _train_num_cols]
            _test_num_display  = [_col_labels.get(c, c) for c in _test_num_cols]

            styled_lb = (
                _lb_df_display.style
                .apply(_hi_max,        subset=_train_num_display)
                .apply(_hi_purple_max, subset=_test_num_display)
                .format(
                    {c: "{:.4f}" for c in _train_num_display + _test_num_display}
                    | {"erosion_prevalence": "{:.2f}%"},
                    na_rep="—",
                )
            )
            st.dataframe(styled_lb, hide_index=True, width="stretch")
            st.caption("Green = best eligible train metric · Purple = best eligible test metric · Grey = excluded (trained on test data)")

            # Bar chart — Global F1 erosion (micro), train vs test
            _bar_has_test = compare_df["test_global_f1"].notna().any()
            _bar_view_opts = ["Train", "Both"] if _bar_has_test else ["Train"]
            if _bar_has_test:
                _bar_view_opts = ["Train", "Test", "Both"]
            _bar_view = st.radio(
                "Bar chart dataset", _bar_view_opts,
                index=1 if _bar_has_test else 0,
                horizontal=True, key="bar_view",
            )

            if _bar_view == "Both":
                _bc1, _bc2 = st.columns(2)
                for _col, _title, _col_scale, _col_container in [
                    ("global_f1_erosion", "Global F1 erosion — Train (micro)", "RdYlGn", _bc1),
                    ("test_global_f1",    "Global F1 erosion — Test (micro)",  "PuRd",   _bc2),
                ]:
                    _bdf = compare_df.sort_values(_col, ascending=True, na_position="first")
                    _col_container.plotly_chart(
                        _bar_grey(_bdf, _col, _title, _col_scale), width='stretch'
                    )
            else:
                _bar_col    = "global_f1_erosion" if _bar_view == "Train" else "test_global_f1"
                _bar_cscale = "RdYlGn" if _bar_view == "Train" else "PuRd"
                _bar_title  = f"Global F1 erosion — {_bar_view} (micro)"
                _bdf = compare_df.sort_values(_bar_col, ascending=True, na_position="first")
                st.plotly_chart(_bar_grey(_bdf, _bar_col, _bar_title, _bar_cscale),
                                width='stretch')

            st.divider()

            # ── Section 2: Distribution comparison ───────────────────────────
            st.markdown("#### 2 · F1 distributions across tiles")

            _has_test_dist = bool(test_tile_data)
            _dist_view_opts = ["Train", "Both (overlay)"] if _has_test_dist else ["Train"]
            if _has_test_dist:
                _dist_view_opts = ["Train", "Test", "Both (overlay)"]
            _cmp_dist_view = st.radio(
                "Dataset", _dist_view_opts,
                index=1 if _has_test_dist else 0,   # default to "Test" when available
                horizontal=True, key="cmp_dist_view",
            )

            _d_filter = st.radio("Tiles", ["All", "Erosion only (n_erosion_pixels > 0)"],
                                 horizontal=True, key="d_filter")
            _d_models = st.multiselect("Models", list(tile_data.keys()),
                                       default=list(tile_data.keys()), key="d_models",
                                       format_func=lambda m: Path(m).stem)

            if _d_models:
                _dist_frames = []
                _show_train_dist = _cmp_dist_view in ("Train", "Both (overlay)")
                _show_test_dist  = _cmp_dist_view in ("Test",  "Both (overlay)") and _has_test_dist

                if _show_train_dist:
                    for _mf in _d_models:
                        _df = tile_data[_mf].copy()
                        if _d_filter.startswith("Erosion"):
                            _df = _df[_df["n_erosion_pixels"] > 0]
                        _df["model"] = Path(_mf).stem
                        _df["set"]   = "Train"
                        _dist_frames.append(_df[["model", "set", "f1_erosion", "f1_no_erosion", "iou_erosion"]])

                if _show_test_dist:
                    for _mf in _d_models:
                        if _mf not in test_tile_data:
                            continue
                        _df = test_tile_data[_mf].copy()
                        if _d_filter.startswith("Erosion"):
                            _df = _df[_df["n_erosion_pixels"] > 0]
                        _df["model"] = Path(_mf).stem
                        _df["set"]   = "Test"
                        _dist_frames.append(_df[["model", "set", "f1_erosion", "f1_no_erosion", "iou_erosion"]])

                if _dist_frames:
                    _dist_all = pd.concat(_dist_frames, ignore_index=True)
                    # When "Both", colour by model; title suffix distinguishes set via shape/opacity
                    _color_col = "model"
                    _box_title  = "F1 erosion — box plot"
                    _hist_title = "F1 erosion — histogram"
                    if _cmp_dist_view == "Both (overlay)":
                        # Encode set into model label so each gets its own colour
                        _dist_all["model_set"] = _dist_all["model"] + " [" + _dist_all["set"] + "]"
                        _color_col = "model_set"
                        _box_title  = "F1 erosion — box plot  (Train · Test)"
                        _hist_title = "F1 erosion — histogram  (Train · Test, overlay)"

                    _dc1, _dc2 = st.columns(2)
                    with _dc1:
                        _fig_box = px.box(
                            _dist_all,
                            x=_color_col if _cmp_dist_view == "Both (overlay)" else "model",
                            y="f1_erosion", color=_color_col,
                            title=_box_title,
                            labels={"f1_erosion": "F1 erosion", _color_col: ""},
                            color_discrete_map=_grey_cmap,
                        )
                        _fig_box.update_layout(showlegend=False, height=420)
                        _fig_box.update_xaxes(tickangle=-25)
                        st.plotly_chart(_fig_box, width='stretch')
                    with _dc2:
                        _fig_hist = px.histogram(
                            _dist_all, x="f1_erosion", color=_color_col,
                            nbins=40, barmode="overlay", opacity=0.55,
                            title=_hist_title,
                            labels={"f1_erosion": "F1 erosion"},
                            color_discrete_map=_grey_cmap,
                        )
                        _fig_hist.update_layout(height=420)
                        st.plotly_chart(_fig_hist, width='stretch')

            st.divider()

            # ── Section 3: Pairwise deep-dive ─────────────────────────────────
            st.markdown("#### 3 · Pairwise deep-dive")
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

            # Train/test toggle for pairwise — default to Test when both have test data
            _pw_has_test = _pw_a in test_tile_data and _pw_b in test_tile_data
            _pw_view_opts = ["Train", "Both"] if _pw_has_test else ["Train"]
            if _pw_has_test:
                _pw_view_opts = ["Train", "Test", "Both"]
            _pw_view = st.radio(
                "Pairwise dataset", _pw_view_opts,
                index=1 if _pw_has_test else 0,   # default to "Test" when available
                horizontal=True, key="pw_view",
            )

            if _pw_a == _pw_b:
                st.warning("Select two different models.")
            else:
                _cols_pw = ["imagery_file", "f1_erosion", "precision_erosion",
                            "recall_erosion", "iou_erosion", "n_erosion_pixels"]
                _stem_a = Path(_pw_a).stem
                _stem_b = Path(_pw_b).stem
                _cdmap  = {
                    f"A wins ({_stem_a})": "#2196F3",
                    f"B wins ({_stem_b})": "#FF5722",
                    "Tie": "#888888",
                }

                def _build_pairwise(src_a, src_b, label):
                    _dfa = src_a[_cols_pw].copy()
                    _dfb = src_b[_cols_pw].copy()
                    m = _dfa.merge(_dfb, on="imagery_file", suffixes=("_a", "_b"))
                    if len(m) == 0:
                        return None
                    m["delta"]  = m["f1_erosion_a"] - m["f1_erosion_b"]
                    m["result"] = m["delta"].apply(
                        lambda d: f"A wins ({_stem_a})" if d > _win_thresh
                                  else (f"B wins ({_stem_b})" if d < -_win_thresh else "Tie")
                    )
                    m["set"] = label
                    return m

                _show_train_pw = _pw_view in ("Train", "Both")
                _show_test_pw  = _pw_view in ("Test",  "Both") and _pw_has_test

                _pw_frames = []
                if _show_train_pw:
                    _m = _build_pairwise(tile_data[_pw_a], tile_data[_pw_b], "Train")
                    if _m is not None:
                        _pw_frames.append(_m)
                if _show_test_pw:
                    _m = _build_pairwise(test_tile_data[_pw_a], test_tile_data[_pw_b], "Test")
                    if _m is not None:
                        _pw_frames.append(_m)

                if not _pw_frames:
                    st.warning("No tiles in common — these models used different datasets.")
                else:
                    for _pw_df in _pw_frames:
                        _set_label = _pw_df["set"].iloc[0]
                        _n_a   = (_pw_df["delta"] >  _win_thresh).sum()
                        _n_b   = (_pw_df["delta"] < -_win_thresh).sum()
                        _n_tie = len(_pw_df) - _n_a - _n_b

                        if _pw_view == "Both":
                            st.markdown(f"**{_set_label} set** — {len(_pw_df):,} tiles in common")
                        else:
                            st.caption(f"**{len(_pw_df):,}** tiles in common")

                        _kc1, _kc2, _kc3 = st.columns(3)
                        _kc1.metric(f"A wins — {_stem_a}", int(_n_a))
                        _kc2.metric("Ties", int(_n_tie))
                        _kc3.metric(f"B wins — {_stem_b}", int(_n_b))

                        _pw_sc_col, _pw_hist_col = st.columns([3, 2])
                        with _pw_sc_col:
                            _fig_pw = px.scatter(
                                _pw_df,
                                x="f1_erosion_a", y="f1_erosion_b", color="result",
                                color_discrete_map=_cdmap,
                                size="n_erosion_pixels_a", size_max=14,
                                hover_data={"imagery_file": True, "delta": ":.4f",
                                            "n_erosion_pixels_a": True},
                                title=f"{_stem_a}  vs  {_stem_b} [{_set_label}]",
                                labels={"f1_erosion_a": f"F1 A ({_stem_a})",
                                        "f1_erosion_b": f"F1 B ({_stem_b})"},
                            )
                            _fig_pw.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                                              line=dict(dash="dot", color="white", width=1))
                            _fig_pw.update_layout(height=480)
                            st.plotly_chart(_fig_pw, width='stretch')
                        with _pw_hist_col:
                            _fig_delta = px.histogram(
                                _pw_df, x="delta", color="result",
                                color_discrete_map=_cdmap,
                                nbins=40, barmode="overlay", opacity=0.7,
                                title=f"Δ F1 distribution (A − B) [{_set_label}]",
                                labels={"delta": "Δ F1 (A − B)", "result": ""},
                            )
                            _fig_delta.add_vline(x=0, line_dash="dot",
                                                 line_color="white", line_width=1)
                            _fig_delta.update_layout(height=480, showlegend=False)
                            st.plotly_chart(_fig_delta, width='stretch')

            st.divider()

            # ── Section 4: Erosion density distribution + gap decomposition ──────
            st.markdown("#### 4 · Erosion density")
            _ref_src = test_tile_data if _has_test_dist else tile_data
            _ref_set_lbl = "test" if _has_test_dist else "train"
            _ref_mf = next(iter(_ref_src), None)
            if _ref_mf is not None:
                _ref = _ref_src[_ref_mf].copy()
                _ref_tot = (_ref["n_erosion_pixels"] + _ref["n_no_erosion_pixels"]).clip(lower=1)
                _ref_pct = _ref["n_erosion_pixels"] / _ref_tot * 100
                _ref["density"] = pd.cut(
                    _ref_pct, bins=[-0.001, 0, 5, 25, 100],
                    labels=["None (0%)", "Sparse (0–5%)", "Medium (5–25%)", "Dense (>25%)"],
                )
                _ref["erosion_pct"] = _ref_pct

                _count_tbl = (
                    _ref.groupby("density", observed=True)
                    .size().reset_index(name="# tiles")
                )
                _total_ref = _count_tbl["# tiles"].sum()
                _count_tbl["% of set"] = (
                    (_count_tbl["# tiles"] / _total_ref * 100).round(1).astype(str) + " %"
                )
                _count_tbl = _count_tbl.rename(columns={"density": "Density bucket"})

                _ctx_l, _ctx_r = st.columns([1, 2])
                with _ctx_l:
                    st.caption(
                        f"Tile count per bucket ({_ref_set_lbl} set · ground truth · model-independent). "
                        "A bucket with far more tiles dominates any weighted mean."
                    )
                    st.dataframe(_count_tbl, hide_index=True, width='stretch')
                with _ctx_r:
                    _ero_only = _ref[_ref["erosion_pct"] > 0]
                    _fig_sparsity = px.histogram(
                        _ero_only, x="erosion_pct", nbins=60,
                        title=f"Erosion coverage distribution — {_ref_set_lbl} tiles with erosion ({len(_ero_only):,})",
                        labels={"erosion_pct": "Erosion pixels (% of tile)"},
                    )
                    _fig_sparsity.update_layout(
                        height=300,
                        xaxis_title="Erosion % of tile",
                        yaxis_title="# tiles",
                        margin=dict(t=40),
                    )
                    st.plotly_chart(_fig_sparsity, width='stretch')

            # ── Mean F1 gap decomposition (test set, pairwise) ──────────
            if len(test_tile_data) >= 2:
                    st.markdown("##### Mean F1 gap decomposition (test set)")
                    _gap_models = list(test_tile_data.keys())
                    _gap_labels = [Path(m).stem for m in _gap_models]
                    _gc1, _gc2 = st.columns(2)
                    _gap_a_idx = _gc1.selectbox(
                        "Model A", _gap_labels, index=0, key="gap_model_a"
                    )
                    _gap_b_idx = _gc2.selectbox(
                        "Model B", _gap_labels,
                        index=min(1, len(_gap_labels) - 1),
                        key="gap_model_b",
                    )
                    _gap_mf_a = _gap_models[_gap_labels.index(_gap_a_idx)]
                    _gap_mf_b = _gap_models[_gap_labels.index(_gap_b_idx)]

                    if _gap_mf_a == _gap_mf_b:
                        st.info("Select two different models.")
                    else:
                        _ga = test_tile_data[_gap_mf_a].copy()
                        _gb = test_tile_data[_gap_mf_b][["imagery_file", "f1_erosion"]].copy()
                        _gm = _ga.merge(_gb, on="imagery_file", suffixes=("_a", "_b"))
                        _gm = _gm[_gm["n_erosion_pixels"] > 0].copy()

                        if _gm.empty:
                            st.warning("No shared erosion tiles found.")
                        else:
                            _g_tot_px = (_gm["n_erosion_pixels"] + _gm["n_no_erosion_pixels"]).clip(lower=1)
                            _g_pct    = _gm["n_erosion_pixels"] / _g_tot_px * 100
                            _gm["bucket"] = pd.cut(
                                _g_pct, bins=[-0.001, 5, 25, 100],
                                labels=["Sparse (0–5%)", "Medium (5–25%)", "Dense (>25%)"],
                            )
                            _G_N = len(_gm)

                            _mf1_a = float(_gm["f1_erosion_a"].mean())
                            _mf1_b = float(_gm["f1_erosion_b"].mean())

                            _col = "Σ(f1_A − f1_B) / N"
                            _gap_rows = []
                            for _bkt in ["Sparse (0–5%)", "Medium (5–25%)", "Dense (>25%)"]:
                                _sub = _gm[_gm["bucket"] == _bkt]
                                _val = float((_sub["f1_erosion_a"] - _sub["f1_erosion_b"]).sum()) / _G_N
                                _gap_rows.append({"Bucket": _bkt, _col: round(_val, 4)})

                            _total_val = _mf1_a - _mf1_b
                            _gap_rows.append({"Bucket": "TOTAL", _col: round(_total_val, 4)})

                            _gap_df_full = pd.DataFrame(_gap_rows)

                            def _style_gap(s: pd.Series) -> list[str]:
                                out = []
                                for i, v in enumerate(s):
                                    try:
                                        fv = float(v)
                                    except (TypeError, ValueError):
                                        out.append("")
                                        continue
                                    bold = "font-weight:bold;" if i == len(s) - 1 else ""
                                    if fv > 0.0005:
                                        out.append(f"{bold}color:#2ecc71")
                                    elif fv < -0.0005:
                                        out.append(f"{bold}color:#e74c3c")
                                    else:
                                        out.append(f"{bold}color:#888")
                                return out

                            st.dataframe(
                                _gap_df_full.style
                                .apply(_style_gap, subset=[_col])
                                .format({_col: "{:+.4f}"}),
                                hide_index=True, width='stretch',
                            )
                            st.caption(
                                f"TOTAL = mean_F1(A) − mean_F1(B) = "
                                f"**{_mf1_a:.4f}** − **{_mf1_b:.4f}** = **{_total_val:+.4f}** · "
                                f"N = {_G_N:,} shared erosion tiles · "
                                f"positive = A better · negative = B better"
                            )

                            # ── Corroboration charts ──────────────────────────
                            _bkt_order = ["Sparse (0–5%)", "Medium (5–25%)", "Dense (>25%)"]
                            _bkt_agg = (
                                _gm.groupby("bucket", observed=True)
                                .agg(
                                    mean_a=("f1_erosion_a", "mean"),
                                    mean_b=("f1_erosion_b", "mean"),
                                    n=("f1_erosion_a", "count"),
                                )
                                .reindex(_bkt_order)
                                .reset_index()
                            )
                            _bkt_agg["contrib"] = (
                                (_bkt_agg["mean_a"] - _bkt_agg["mean_b"])
                                * _bkt_agg["n"] / _G_N
                            )

                            _corr_l, _corr_r = st.columns(2)

                            # Left: mean F1 per bucket side-by-side
                            _bkt_long = pd.concat([
                                _bkt_agg[["bucket", "mean_a"]].rename(columns={"mean_a": "mean_f1"}).assign(model=_gap_a_idx),
                                _bkt_agg[["bucket", "mean_b"]].rename(columns={"mean_b": "mean_f1"}).assign(model=_gap_b_idx),
                            ])
                            _fig_corr_means = px.bar(
                                _bkt_long, x="bucket", y="mean_f1", color="model",
                                barmode="group", text_auto=".3f",
                                title="Mean F1 per bucket (decomposition tiles)",
                                labels={"bucket": "Bucket", "mean_f1": "Mean F1 erosion"},
                                category_orders={"bucket": _bkt_order},
                                color_discrete_sequence=["#3498db", "#e67e22"],
                            )
                            _fig_corr_means.update_layout(height=320, margin=dict(t=40))
                            _corr_l.plotly_chart(_fig_corr_means, width='stretch')

                            # Right: contribution bars (should match table values)
                            _bkt_agg["color"] = _bkt_agg["contrib"].apply(
                                lambda v: "#2ecc71" if v > 0.0005 else ("#e74c3c" if v < -0.0005 else "#888888")
                            )
                            _fig_corr_contrib = px.bar(
                                _bkt_agg, x="bucket", y="contrib",
                                text=_bkt_agg["contrib"].apply(lambda v: f"{v:+.4f}"),
                                title="Contributions — Σ(f1_A−f1_B)/N per bucket",
                                labels={"bucket": "Bucket", "contrib": "Contribution"},
                                category_orders={"bucket": _bkt_order},
                                color="color",
                                color_discrete_map="identity",
                            )
                            _fig_corr_contrib.update_layout(
                                height=320, margin=dict(t=40), showlegend=False,
                            )
                            _fig_corr_contrib.add_hline(y=0, line_width=1, line_color="gray")
                            _corr_r.plotly_chart(_fig_corr_contrib, width='stretch')


# ── TAB 5: Test set ───────────────────────────────────────────────────────────
with tab_test:
    st.subheader("Test set explorer")

    _tmet_path = _test_metrics_path(model_stem)

    if _tmet_path is None:
        st.info(
            f"No test metrics found for **{selected_model_name}**.  \n"
            "Run the evaluation pipeline first:  \n"
            "```\npython -m DASHBOARD_FROM_TYTONAI.evaluate_test\n```"
        )
    else:
        tcon = get_test_con(str(_tmet_path))

        def tq(query: str) -> pd.DataFrame:
            return tcon.execute(query).df()

        _t_has_cm = "tp_erosion" in tcon.execute(
            "SELECT * FROM test_metrics LIMIT 0"
        ).df().columns

        # ── Load test geo for capture filter ─────────────────────────────────
        _tgeo = _test_geo_path()
        test_geo_df = pd.read_parquet(_tgeo) if _tgeo.exists() else pd.DataFrame()

        # ── KPIs ──────────────────────────────────────────────────────────────
        _t_cm_expr = (
            ", SUM(tp_erosion) AS sum_tp, SUM(fp_erosion) AS sum_fp, SUM(fn_erosion) AS sum_fn"
            if _t_has_cm else ""
        )
        tkpi = tq(f"""
            SELECT
                COUNT(*)                                                        AS n_tiles,
                COUNTIF(n_erosion_pixels > 0)                                   AS n_ero_tiles,
                AVG(CASE WHEN n_erosion_pixels > 0 THEN f1_erosion        END)  AS f1_mean,
                AVG(CASE WHEN n_erosion_pixels > 0 THEN precision_erosion END)  AS prec_mean,
                AVG(CASE WHEN n_erosion_pixels > 0 THEN recall_erosion    END)  AS rec_mean
                {_t_cm_expr}
            FROM test_metrics
        """).iloc[0]

        if _t_has_cm:
            _ttp = float(tkpi["sum_tp"]); _tfp = float(tkpi["sum_fp"]); _tfn = float(tkpi["sum_fn"])
            _td  = 2 * _ttp + _tfp + _tfn
            t_global_f1 = (2 * _ttp / _td) if _td > 0 else float("nan")
        else:
            t_global_f1 = None

        def _purple_kpi(label: str, value: str, sub: str = "") -> str:
            return (
                f'<div style="background:#1a0a2a;border-radius:8px;padding:10px 4px;'
                f'text-align:center;margin:2px;border:1px solid #4a235a">'
                f'<div style="color:#bbb;font-size:0.78em;margin-bottom:4px">{label}</div>'
                f'<div style="color:#9B59B6;font-size:1.5em;font-weight:700">{value}</div>'
                f'<div style="color:#888;font-size:0.72em">{sub}</div>'
                f'</div>'
            )

        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.markdown(_purple_kpi("Tiles", f"{int(tkpi['n_tiles']):,}", "test set"),
                     unsafe_allow_html=True)
        tc2.markdown(_purple_kpi("Erosion tiles", f"{int(tkpi['n_ero_tiles']):,}"),
                     unsafe_allow_html=True)
        tc3.markdown(_purple_kpi("Global F1", _fmt(t_global_f1) if t_global_f1 is not None else "—",
                                 "micro"),
                     unsafe_allow_html=True)
        tc4.markdown(_purple_kpi("Mean F1", _fmt(tkpi["f1_mean"]), "erosion tiles only"),
                     unsafe_allow_html=True)
        tc5.markdown(_purple_kpi("Recall", _fmt(tkpi["rec_mean"])), unsafe_allow_html=True)

        st.divider()

        # ── Filters ───────────────────────────────────────────────────────────
        tf1, tf2, tf3, tf4 = st.columns([2, 2, 2, 2])
        with tf1:
            _t_sort_opts = {
                "★ Worst F1 erosion":             "f1_erosion ASC, n_erosion_pixels DESC",
                "★ Best F1 erosion":              "f1_erosion DESC, n_erosion_pixels DESC",
                "★ Worst recall (miss erosion)":  "recall_erosion ASC, n_erosion_pixels DESC",
                "★ Worst precision (false alarm)":"precision_erosion ASC, n_erosion_pixels DESC",
                "Most GT erosion px":             "n_erosion_pixels DESC",
                "F1 erosion ↓ (best first)":      "f1_erosion DESC",
                "F1 erosion ↑ (worst first)":     "f1_erosion ASC",
            }
            t_sort_label   = st.selectbox("Sort by", list(_t_sort_opts.keys()), index=0,
                                          key="t_sort")
            t_order_clause = _t_sort_opts[t_sort_label]
        with tf2:
            t_f1_range = st.slider("F1 erosion range", 0.0, 1.0, (0.0, 1.0),
                                   step=0.01, key="t_f1_range")
        with tf3:
            _t_max_px    = int(tq("SELECT MAX(n_erosion_pixels) FROM test_metrics").iloc[0, 0])
            t_min_ero_px = st.number_input("Min GT erosion px", 0, _t_max_px, 0, step=100,
                                           key="t_min_px")
        with tf4:
            _capture_names = (
                sorted(test_geo_df["capture_name"].dropna().unique().tolist())
                if not test_geo_df.empty and "capture_name" in test_geo_df.columns
                else []
            )
            if _capture_names:
                selected_captures = st.multiselect(
                    "Filter by capture", _capture_names, default=_capture_names,
                    key="t_captures",
                )
            else:
                selected_captures = []

        # Join with geo to apply capture filter
        _t_join = ""
        _t_where_capture = ""
        if selected_captures and not test_geo_df.empty and _tgeo.exists():
            _cap_list = ", ".join(f"'{c}'" for c in selected_captures)
            _t_join = f"JOIN read_parquet('{_tgeo}') g ON t.imagery_file = g.imagery_file"
            _t_where_capture = f"AND g.capture_name IN ({_cap_list})"

        _pred_col = "(t.tp_erosion + t.fp_erosion) AS pred_ero_px" if _t_has_cm else "NULL AS pred_ero_px"
        t_filtered = tcon.execute(f"""
            SELECT t.imagery_file, t.mask_file,
                   ROUND(t.f1_erosion, 4)        AS f1_erosion,
                   ROUND(t.precision_erosion, 4) AS precision,
                   ROUND(t.recall_erosion, 4)    AS recall,
                   t.n_erosion_pixels             AS gt_ero_px,
                   t.n_no_erosion_pixels          AS gt_no_ero_px,
                   {_pred_col}
            FROM test_metrics t
            {_t_join}
            WHERE t.f1_erosion BETWEEN {t_f1_range[0]} AND {t_f1_range[1]}
              AND t.n_erosion_pixels >= {t_min_ero_px}
              {_t_where_capture}
            ORDER BY {t_order_clause}
        """).df()

        # Search
        t_search = st.text_input("Search tile name", value="", placeholder="e.g. 3212e040",
                                 key="t_search")
        if t_search:
            t_filtered = t_filtered[t_filtered["imagery_file"].str.contains(
                t_search, case=False, na=False)]

        _t_total = len(t_filtered)
        _t_page_size = st.selectbox("Rows / page", [20, 50, 100], index=0, key="t_page_size")
        _t_pages     = max(1, (_t_total - 1) // _t_page_size + 1)
        tp1, tp2 = st.columns([4, 1])
        with tp1:
            st.caption(f"**{_t_total:,}** tiles match filters")
        with tp2:
            t_page = st.number_input("Page", 1, _t_pages, 1, key="t_page")

        t_page_df = t_filtered.iloc[(t_page - 1) * _t_page_size : t_page * _t_page_size]

        t_event = st.dataframe(
            t_page_df, width='stretch', hide_index=True,
            on_select="rerun", selection_mode="single-row", key="test_table",
        )

        st.divider()

        # ── Click-to-visualise ────────────────────────────────────────────────
        t_sel = t_event.selection.rows if t_event.selection else []
        if not t_sel:
            st.info("Click a row in the table above to view the tile.")
        else:
            t_imagery = t_page_df.iloc[t_sel[0]]["imagery_file"]
            t_mask    = t_page_df.iloc[t_sel[0]]["mask_file"]

            st.subheader("Selected test tile")

            tnc1, tnc2, tnc3 = st.columns([3, 3, 2])
            with tnc1:
                st.markdown("**Imagery file**")
                st.code(t_imagery, language=None)
            with tnc2:
                st.markdown("**Mask file**")
                st.code(t_mask, language=None)
            with tnc3:
                if not test_geo_df.empty:
                    _tgeo_row = test_geo_df[test_geo_df["imagery_file"] == t_imagery]
                    if not _tgeo_row.empty:
                        r = _tgeo_row.iloc[0]
                        st.markdown("**Location**")
                        st.metric("Lat", f"{float(r['lat']):.5f}")
                        st.metric("Lon", f"{float(r['lon']):.5f}")
                        if "capture_name" in r:
                            st.caption(f"Capture: {r['capture_name']}")

            t_full = tq(
                f"SELECT * FROM test_metrics WHERE imagery_file = '{t_imagery}' LIMIT 1"
            ).iloc[0].to_dict()

            with st.spinner("Generating visualisation (cached after first run)…"):
                try:
                    tp1_png, tp2_png = _generate_test_pngs(t_imagery, t_mask, t_full)
                except Exception as e:
                    st.error(f"Generation failed: {e}")
                    _ts = f"test_{model_stem}"
                    tp1_png = _png_path(t_imagery, "masks",   _ts)
                    tp2_png = _png_path(t_imagery, "overlay", _ts)

            t_visu = st.radio(
                "Visualisation", ["Side-by-side masks", "Contour overlay (RGB / DSM)"],
                horizontal=True, key="t_visu_mode",
            )
            _, t_img_col, _ = st.columns([2, 6, 2])
            with t_img_col:
                p = tp1_png if t_visu == "Side-by-side masks" else tp2_png
                if p.exists():
                    st.image(str(p), width="stretch")
                else:
                    missing_dir = _TEST_DATA_DIR
                    st.warning(
                        f"NPZ not found in `{missing_dir}`.  \n"
                        "Run `python -m DASHBOARD_FROM_TYTONAI.evaluate_test` to download tiles."
                    )

            _show_tile_metrics(t_full)


# ── TAB 6: Raw data ───────────────────────────────────────────────────────────
with tab_data:
    st.subheader("Raw data")

    st.markdown("**Training metrics — first 5 000 rows (sorted by F1 erosion ↑)**")
    st.dataframe(
        q("SELECT * FROM metrics ORDER BY f1_erosion ASC LIMIT 5000"),
        width="stretch", hide_index=True, height=420,
    )

    _tmet_raw = _test_metrics_path(model_stem)
    if _tmet_raw:
        st.divider()
        st.markdown("**Test metrics — first 5 000 rows (sorted by F1 erosion ↑)**")
        _traw = pd.read_parquet(_tmet_raw).sort_values("f1_erosion").head(5000)
        st.dataframe(_traw, width="stretch", hide_index=True, height=420)

    st.divider()
    st.markdown("**DuckDB SQL console**")
    st.caption(
        "Available views: `metrics` (train set), `test_metrics` (test set — if evaluated).  \n"
        "Example: `SELECT * FROM metrics WHERE f1_erosion < 0.1 LIMIT 20`"
    )
    user_sql = st.text_area(
        "SQL",
        value="SELECT * FROM metrics ORDER BY f1_erosion ASC LIMIT 20",
        height=160,
    )
    if st.button("Run"):
        try:
            # Register test_metrics in the same console if available
            if _tmet_raw:
                con.execute(f"""
                    CREATE OR REPLACE VIEW test_metrics AS
                    SELECT * FROM read_parquet('{_tmet_raw}')
                """)
            res = q(user_sql)
            st.success(f"{len(res):,} rows")
            st.dataframe(res, width="stretch", hide_index=True)
        except Exception as e:
            st.error(str(e))
