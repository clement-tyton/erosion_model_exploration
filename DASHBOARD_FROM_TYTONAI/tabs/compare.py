"""
Tab 4 — Compare Models

Leaderboard (train + test), F1 distributions, pairwise deep-dive, and erosion
density gap decomposition across all models in the registry.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render(
    registry: dict[str, dict],
    registry_path: Path,
    output_dir: Path,
) -> None:

    st.subheader("Model comparison")

    if not registry_path.exists():
        st.warning("No `models_registry.json` found at repo root.")
        return

    @st.cache_data(show_spinner="Loading aggregate metrics…")
    def _load_all_metrics(registry_hash: str) -> pd.DataFrame:
        rows = []
        for entry in list(registry.values()):
            stem = Path(entry["model_file"]).stem
            pq = next(
                (p for p in [output_dir / f"metrics_{stem}.parquet",
                              output_dir / "metrics.parquet"] if p.exists()),
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

            # ── Test set metrics (if available) ───────────────────────────────
            tpq = output_dir / f"test_metrics_{stem}.parquet"
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
        for entry in list(registry.values()):
            stem = Path(entry["model_file"]).stem
            pq = next(
                (p for p in [output_dir / f"metrics_{stem}.parquet",
                              output_dir / "metrics.parquet"] if p.exists()),
                None,
            )
            if pq is None:
                continue
            out[entry["model_file"]] = pd.read_parquet(pq)
        return out

    @st.cache_data(show_spinner="Loading test tile-level data…")
    def _load_test_tile_data(registry_hash: str) -> dict[str, pd.DataFrame]:
        out = {}
        for entry in list(registry.values()):
            stem = Path(entry["model_file"]).stem
            tpq  = output_dir / f"test_metrics_{stem}.parquet"
            if not tpq.exists():
                continue
            out[entry["model_file"]] = pd.read_parquet(tpq)
        return out

    _reg_hash      = str(hash(registry_path.read_text()))
    compare_df     = _load_all_metrics(_reg_hash)
    tile_data      = _load_tile_data(_reg_hash)
    test_tile_data = _load_test_tile_data(_reg_hash)

    if compare_df.empty:
        st.info("No metrics computed yet. Run `python -m src.run_all` to evaluate.")
        return

    n_evaluated = len(compare_df)
    n_total     = len(registry)
    n_test      = compare_df["test_global_f1"].notna().sum()
    st.caption(
        f"**{n_evaluated} / {n_total}** models evaluated "
        f"({n_test} with test-set metrics)"
    )

    # v2 was trained on the test set — show its results but exclude from winner/highlight
    _EXCLUDED = {"v2"}
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

    # ── Section 1: Leaderboard ─────────────────────────────────────────────────
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

    _is_excluded = compare_df.set_index("label").reindex(
        _lb_df["label"]
    )["version"].isin(_EXCLUDED).values

    _v2_stems = set(compare_df[compare_df["version"].isin(_EXCLUDED)]["label"])
    _GREY = "#888888"
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
        st.plotly_chart(_bar_grey(_bdf, _bar_col, _bar_title, _bar_cscale), width='stretch')

    st.divider()

    # ── Section 2: Distribution comparison ────────────────────────────────────
    st.markdown("#### 2 · F1 distributions across tiles")

    _has_test_dist = bool(test_tile_data)
    _dist_view_opts = ["Train", "Both (overlay)"] if _has_test_dist else ["Train"]
    if _has_test_dist:
        _dist_view_opts = ["Train", "Test", "Both (overlay)"]
    _cmp_dist_view = st.radio(
        "Dataset", _dist_view_opts,
        index=1 if _has_test_dist else 0,
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
            _color_col = "model"
            _box_title  = "F1 erosion — box plot"
            _hist_title = "F1 erosion — histogram"
            if _cmp_dist_view == "Both (overlay)":
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

    # ── Section 3: Pairwise deep-dive ─────────────────────────────────────────
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

    _pw_has_test = _pw_a in test_tile_data and _pw_b in test_tile_data
    _pw_view_opts = ["Train", "Both"] if _pw_has_test else ["Train"]
    if _pw_has_test:
        _pw_view_opts = ["Train", "Test", "Both"]
    _pw_view = st.radio(
        "Pairwise dataset", _pw_view_opts,
        index=1 if _pw_has_test else 0,
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

    # ── Section 4: Erosion density distribution + gap decomposition ───────────
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

    # ── Mean F1 gap decomposition (test set, pairwise) ────────────────────────
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

                # ── Corroboration charts ──────────────────────────────────────
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

    st.divider()

    # ── Section 5: Lorenz-style comparison curves ──────────────────────────────
    st.markdown("#### 5 · Lorenz-style performance curves — model comparison")
    st.caption(
        "Tiles are sorted by **ascending erosion pixel count** (poorest → richest). "
        "At each point N the curves show the global (micro) metric on the first N tiles. "
        "Only tiles with n_erosion_pixels > 0 are included."
    )

    _curve_src_opts = ["Train", "Test"] if bool(test_tile_data) else ["Train"]
    _curve_src = st.radio(
        "Dataset for curves", _curve_src_opts,
        index=1 if bool(test_tile_data) else 0,
        horizontal=True, key="curves_src",
    )
    _curve_pool = test_tile_data if _curve_src == "Test" else tile_data

    _curve_models = st.multiselect(
        "Models to overlay", list(_curve_pool.keys()),
        default=list(_curve_pool.keys()),
        key="curve_models",
        format_func=lambda m: Path(m).stem,
    )

    @st.cache_data(show_spinner=False)
    def _lorenz_curves_for(model_file: str, src: str) -> pd.DataFrame | None:
        pool = test_tile_data if src == "Test" else tile_data
        if model_file not in pool:
            return None
        df = pool[model_file].copy()
        df = df[df["n_erosion_pixels"] > 0].sort_values(
            "n_erosion_pixels", ascending=True
        ).reset_index(drop=True)
        if len(df) == 0:
            return None
        n = len(df)
        x = (np.arange(1, n + 1) / n) * 100
        total_ero = df["n_erosion_pixels"].sum()
        lorenz_y  = df["n_erosion_pixels"].cumsum() / total_ero * 100
        cum_tp    = df["tp_erosion"].cumsum()
        cum_fp    = df["fp_erosion"].cumsum()
        cum_fn    = df["fn_erosion"].cumsum()
        prec = cum_tp / (cum_tp + cum_fp).replace(0, np.nan) * 100
        rec  = cum_tp / (cum_tp + cum_fn).replace(0, np.nan) * 100
        f1   = 2 * prec * rec / (prec + rec).replace(0, np.nan)
        return pd.DataFrame({
            "x": x,
            "lorenz_y": lorenz_y.values,
            "precision": prec.values,
            "recall": rec.values,
            "f1": f1.values,
        })

    if _curve_models:
        _COLORS = [
            "#4a9eda", "#f0983a", "#3acf7a", "#c85ab4",
            "#e05252", "#f5d142", "#52c4c4", "#a0a0ff",
            "#ff8c69", "#90ee90",
        ]

        def _comparison_fig(metric: str, title: str) -> go.Figure:
            fig = go.Figure()
            for i, mf in enumerate(_curve_models):
                cv = _lorenz_curves_for(mf, _curve_src)
                if cv is None:
                    continue
                stem = Path(mf).stem
                color = _COLORS[i % len(_COLORS)]
                is_excl = Path(mf).stem.replace("model_", "").split("_")[0] in _EXCLUDED
                opacity = 0.35 if is_excl else 0.9
                dash    = "dot" if is_excl else "solid"
                fig.add_trace(go.Scatter(
                    x=cv["x"], y=cv[metric],
                    mode="lines", name=stem,
                    line=dict(color=color, width=2, dash=dash),
                    opacity=opacity,
                    hovertemplate=f"{stem}<br>Tiles: %{{x:.1f}}%<br>{title}: %{{y:.1f}}%<extra></extra>",
                ))
            if metric == "lorenz_y":
                fig.add_trace(go.Scatter(
                    x=[0, 100], y=[0, 100], mode="lines",
                    name="Perfect equality",
                    line=dict(color="white", width=1, dash="dot"),
                    showlegend=True,
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Cumulative % of tiles (sorted by ascending erosion pixel count)",
                yaxis_title=title,
                xaxis=dict(range=[0, 100], ticksuffix="%"),
                yaxis=dict(range=[0, 102], ticksuffix="%"),
                height=420,
                legend=dict(orientation="h", y=-0.25, x=0),
            )
            return fig

        st.plotly_chart(_comparison_fig("lorenz_y", "Lorenz — Cumulative erosion pixels"), width="stretch")

        _cmp_c1, _cmp_c2 = st.columns(2)
        with _cmp_c1:
            st.plotly_chart(_comparison_fig("precision", "Global Precision (erosion)"), width="stretch")
        with _cmp_c2:
            st.plotly_chart(_comparison_fig("recall", "Global Recall (erosion)"), width="stretch")
        st.plotly_chart(_comparison_fig("f1", "Global F1 score (erosion)"), width="stretch")
