"""
Tab 5 — Test Set

Interactive explorer for test tiles: filter/sort, click-to-visualise, and
per-tile metric display. Mirrors Tab 2 (Tile Explorer) but for the test set.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import streamlit as st

from tabs.helpers import fmt, show_tile_metrics


def render(
    model_stem: str,
    selected_model_name: str,
    generate_test_pngs_fn: Callable,
    png_path_fn: Callable,
    test_metrics_path_fn: Callable[[str], Path | None],
    get_test_con_fn: Callable,
    test_geo_path_fn: Callable[[], Path],
    test_data_dir: Path,
) -> None:

    st.subheader("Test set explorer")

    _tmet_path = test_metrics_path_fn(model_stem)

    if _tmet_path is None:
        st.info(
            f"No test metrics found for **{selected_model_name}**.  \n"
            "Run the evaluation pipeline first:  \n"
            "```\npython -m DASHBOARD_FROM_TYTONAI.evaluate_test\n```"
        )
        return

    tcon = get_test_con_fn(str(_tmet_path))

    def tq(query: str) -> pd.DataFrame:
        return tcon.execute(query).df()

    _t_has_cm = "tp_erosion" in tcon.execute(
        "SELECT * FROM test_metrics LIMIT 0"
    ).df().columns

    # ── Load test geo for capture filter ──────────────────────────────────────
    _tgeo = test_geo_path_fn()
    test_geo_df = pd.read_parquet(_tgeo) if _tgeo.exists() else pd.DataFrame()

    # ── KPIs ──────────────────────────────────────────────────────────────────
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
    tc3.markdown(_purple_kpi("Global F1", fmt(t_global_f1) if t_global_f1 is not None else "—",
                             "micro"),
                 unsafe_allow_html=True)
    tc4.markdown(_purple_kpi("Mean F1", fmt(tkpi["f1_mean"]), "erosion tiles only"),
                 unsafe_allow_html=True)
    tc5.markdown(_purple_kpi("Recall", fmt(tkpi["rec_mean"])), unsafe_allow_html=True)

    st.divider()

    # ── Filters ───────────────────────────────────────────────────────────────
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

    # ── Click-to-visualise ────────────────────────────────────────────────────
    t_sel = t_event.selection.rows if t_event.selection else []
    if not t_sel:
        st.info("Click a row in the table above to view the tile.")
        return

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
            tp1_png, tp2_png = generate_test_pngs_fn(t_imagery, t_mask, t_full)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            _ts = f"test_{model_stem}"
            tp1_png = png_path_fn(t_imagery, "masks",   subdir=_ts)
            tp2_png = png_path_fn(t_imagery, "overlay", subdir=_ts)

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
            st.warning(
                f"NPZ not found in `{test_data_dir}`.  \n"
                "Run `python -m DASHBOARD_FROM_TYTONAI.evaluate_test` to download tiles."
            )

    show_tile_metrics(t_full)
