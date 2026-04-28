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

Module layout
-------------
services/registry.py  — path helpers, model registry
services/cache.py     — @st.cache_resource / @st.cache_data functions
services/inference.py — PNG generation for train and test tiles
tabs/overview.py      — Tab 1
tabs/explorer.py      — Tab 2
tabs/map.py           — Tab 3
tabs/compare.py       — Tab 4
tabs/test_set.py      — Tab 5
tabs/raw_data.py      — Tab 6
tabs/helpers.py       — shared UI helpers (fmt, show_tile_metrics)
"""
from __future__ import annotations

from functools import partial

import streamlit as st

# ── Services ──────────────────────────────────────────────────────────────────
from src.config import OUTPUT_DIR
from services.registry import (
    MODELS_DIR,
    _REGISTRY,
    _REGISTRY_PATH,
    _TEST_DATA_DIR,
    _metrics_path,
    _model_data_dir,
    _model_color,
    _model_sort_key,
    _registry_entry,
    _test_geo_path,
    _test_metrics_path,
)
from services.cache import get_con, get_model, get_test_con, tile_map
from services.inference import generate_pngs, generate_test_pngs, png_path

# ── Tabs ──────────────────────────────────────────────────────────────────────
from tabs import overview, explorer, map as map_tab, compare, test_set, raw_data, statistics

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Erosion tile explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Model discovery ───────────────────────────────────────────────────────────
_model_files = (
    sorted(MODELS_DIR.glob("*.pth"), key=_model_sort_key)
    if MODELS_DIR.exists() else []
)

if not _model_files:
    st.error(f"No .pth model files found in `{MODELS_DIR}`.")
    st.stop()

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

# ── Model colour badge ────────────────────────────────────────────────────────
_entry = _registry_entry(selected_model_name)
_color = _model_color(selected_model_name)
_version_label = _entry.get("version", "unknown")
_description   = _entry.get("description", selected_model_name)
st.markdown(
    f'<span style="background:{_color};color:#fff;padding:3px 10px;border-radius:12px;'
    f'font-size:0.82em;font-weight:600">{_version_label}</span>'
    f'&nbsp;<span style="color:#aaa;font-size:0.85em">{_description}</span>',
    unsafe_allow_html=True,
)

if metrics_file is None:
    st.warning(
        f"No metrics for **{selected_model_name}** — run:  \n"
        f"`python -m src.evaluate --model-path models/{selected_model_name}`"
    )

# ── DuckDB connection (keyed by metrics file path) ────────────────────────────
if metrics_file is None:
    st.stop()

con = get_con(str(metrics_file))

# ── Resolve session-specific state ────────────────────────────────────────────
_model, _device   = get_model(str(selected_model_path))
_tile_map_dict    = tile_map(selected_model_name)
_data_dir         = _model_data_dir(selected_model_name)

# ── Bind inference functions to session state ─────────────────────────────────
generate_pngs_fn = partial(
    generate_pngs,
    model=_model, device=_device,
    tile_map_dict=_tile_map_dict, data_dir=_data_dir,
    model_stem=model_stem,
)

generate_test_pngs_fn = partial(
    generate_test_pngs,
    model=_model, device=_device,
    model_stem=model_stem, test_data_dir=_TEST_DATA_DIR,
)

png_path_fn = partial(png_path, model_stem=model_stem)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_stats, tab_overview, tab_explorer, tab_map, tab_compare, tab_test, tab_data = st.tabs(
    ["Statistics", "Overview", "Tile explorer", "Map", "Compare models", "Test set", "Raw data"]
)

with tab_stats:
    statistics.render(
        metrics_file=metrics_file,
        model_stem=model_stem,
    )

with tab_overview:
    overview.render(
        con=con,
        model_stem=model_stem,
        test_metrics_path_fn=_test_metrics_path,
    )

with tab_explorer:
    explorer.render(
        con=con,
        model_stem=model_stem,
        selected_model_name=selected_model_name,
        generate_pngs_fn=generate_pngs_fn,
        generate_test_pngs_fn=generate_test_pngs_fn,
        png_path_fn=png_path_fn,
        test_metrics_path_fn=_test_metrics_path,
        get_test_con_fn=get_test_con,
        registry_entry_fn=_registry_entry,
    )

with tab_map:
    map_tab.render(
        con=con,
        model_stem=model_stem,
        selected_model_name=selected_model_name,
        generate_pngs_fn=generate_pngs_fn,
        generate_test_pngs_fn=generate_test_pngs_fn,
        png_path_fn=png_path_fn,
        test_metrics_path_fn=_test_metrics_path,
        metrics_path_fn=_metrics_path,
        registry_entry_fn=_registry_entry,
        test_geo_path_fn=_test_geo_path,
        metrics_file=metrics_file,
    )

with tab_compare:
    compare.render(
        registry=_REGISTRY,
        registry_path=_REGISTRY_PATH,
        output_dir=OUTPUT_DIR,
    )

with tab_test:
    test_set.render(
        model_stem=model_stem,
        selected_model_name=selected_model_name,
        generate_test_pngs_fn=generate_test_pngs_fn,
        png_path_fn=png_path_fn,
        test_metrics_path_fn=_test_metrics_path,
        get_test_con_fn=get_test_con,
        test_geo_path_fn=_test_geo_path,
        test_data_dir=_TEST_DATA_DIR,
    )

with tab_data:
    raw_data.render(
        con=con,
        model_stem=model_stem,
        test_metrics_path_fn=_test_metrics_path,
    )
