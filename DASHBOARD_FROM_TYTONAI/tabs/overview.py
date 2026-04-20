"""
Tab 1 — Overview

Global KPIs + metric distributions for the training set (with optional test set
comparison). Call render() from within a `with tab_overview:` block.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


def render(
    con: duckdb.DuckDBPyConnection,
    model_stem: str,
    test_metrics_path_fn: Callable[[str], Path | None],
) -> None:

    def q(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    _has_cm = "tp_erosion" in con.execute("SELECT * FROM metrics LIMIT 0").df().columns

    _kpi_df = q(f"""
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
    """)
    if _kpi_df.empty:
        st.error("Metrics table is empty — re-run evaluation for this model.")
        return
    kpi = _kpi_df.iloc[0]

    if _has_cm:
        _tp, _fp, _fn = float(kpi["sum_tp"]), float(kpi["sum_fp"]), float(kpi["sum_fn"])
        _denom = 2 * _tp + _fp + _fn
        global_f1 = (2 * _tp / _denom) if _denom > 0 else float("nan")
    else:
        global_f1 = None

    # ── Test KPIs (computed early so they can be shown alongside train KPIs) ──
    _ov_tmet_early = test_metrics_path_fn(model_stem)
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
    _ov_tmet = test_metrics_path_fn(model_stem)
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
