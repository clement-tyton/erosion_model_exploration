"""
Tab 2 — Tile Explorer

Filterable / sortable table of training tiles with click-to-visualise.
The top expander shows the test set tiles for the same model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import streamlit as st

from tabs.helpers import fmt, show_tile_metrics


def render(
    con: duckdb.DuckDBPyConnection,
    model_stem: str,
    selected_model_name: str,
    generate_pngs_fn: Callable,
    generate_test_pngs_fn: Callable,
    png_path_fn: Callable,
    test_metrics_path_fn: Callable[[str], Path | None],
    get_test_con_fn: Callable,
    registry_entry_fn: Callable[[str], dict],
) -> None:

    def q(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    st.subheader("Tile explorer")

    # ── Test set section (shown first) ────────────────────────────────────────
    with st.expander("Test set tiles", expanded=True):
        _exp_tmet = test_metrics_path_fn(model_stem)
        if _exp_tmet is None:
            st.info(
                f"No test metrics for **{selected_model_name}**.  \n"
                "Run: `python -m DASHBOARD_FROM_TYTONAI.evaluate_test` then refresh."
            )
        else:
            _exp_tcon = get_test_con_fn(str(_exp_tmet))

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
            _ekc3.metric("Mean F1 erosion", fmt(float(_exp_kpi['f1_mean'])))
            _ekc4.metric("Recall erosion",  fmt(float(_exp_kpi['rec_mean'])))

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
                        _etp1, _etp2 = generate_test_pngs_fn(_exp_t_img, _exp_t_mask, _exp_t_full)
                    except Exception as _e:
                        st.error(f"Generation failed: {_e}")
                        _etp1 = png_path_fn(_exp_t_img, "masks",   subdir=f"test_{model_stem}")
                        _etp2 = png_path_fn(_exp_t_img, "overlay", subdir=f"test_{model_stem}")

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
                show_tile_metrics(_exp_t_full)

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
            _geo_f = _geo_path_exp(registry_entry_fn(selected_model_name).get("dataset_name", "default"))
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
                p1, p2 = generate_pngs_fn(imagery_file, full)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                p1, p2 = png_path_fn(imagery_file, "masks"), png_path_fn(imagery_file, "overlay")

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

        show_tile_metrics(full)
