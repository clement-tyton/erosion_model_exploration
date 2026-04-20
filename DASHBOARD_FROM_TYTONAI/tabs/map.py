"""
Tab 3 — Map

Geospatial scatter map of tiles coloured by metric, with optional purple test
set overlay and click-to-visualise detail panel.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tabs.helpers import show_tile_metrics


def render(
    con: duckdb.DuckDBPyConnection,
    model_stem: str,
    selected_model_name: str,
    generate_pngs_fn: Callable,
    generate_test_pngs_fn: Callable,
    png_path_fn: Callable,
    test_metrics_path_fn: Callable[[str], Path | None],
    metrics_path_fn: Callable[[str], Path | None],
    registry_entry_fn: Callable[[str], dict],
    test_geo_path_fn: Callable[[], Path],
    metrics_file: Path,
) -> None:
    from src.config import OUTPUT_DIR
    from src.build_geo import geo_parquet_path as _geo_path

    def q(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    st.subheader("Tile map — Western Australia (EPSG 20350 → WGS84)")

    _entry      = registry_entry_fn(selected_model_name)
    _dataset    = _entry.get("dataset_name", "default") if _entry else "default"
    GEO_PARQUET = _geo_path(_dataset)

    if not GEO_PARQUET.exists():
        st.warning(
            f"Geographic index not built yet for dataset **{_dataset}**. Run:\n"
            f"```\npython -m src.run_all\n```\nto build it automatically."
        )
        return

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

    _map_metrics = metrics_path_fn(model_stem)
    if _map_metrics and _map_metrics.suffix == ".parquet":
        map_df_full = load_map_data(str(GEO_PARQUET), str(_map_metrics))
    elif _map_metrics and _map_metrics.suffix == ".csv":
        _tmp = OUTPUT_DIR / "_tmp_metrics.parquet"
        if not _tmp.exists():
            pd.read_csv(_map_metrics).to_parquet(_tmp, index=False)
        map_df_full = load_map_data(str(GEO_PARQUET), str(_tmp))
    else:
        st.error("No metrics file found for the selected model.")
        return

    if map_df_full.empty:
        return

    # ── Controls row ──────────────────────────────────────────────────────────
    map_ctrl1, map_ctrl2 = st.columns([6, 2])
    with map_ctrl1:
        jump_str = st.text_input(
            "Jump to tile (paste name or UUID fragment from Explorer)",
            value="", placeholder="e.g. 16413f59", key="map_jump",
        )
    with map_ctrl2:
        _tgeo    = test_geo_path_fn()
        _tmet    = test_metrics_path_fn(model_stem)
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

    # ── Load test tile data for overlay ───────────────────────────────────────
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
        return

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
    fig_map.data[0].update(marker=dict(opacity=0.95))

    # ── Jump highlight overlay ─────────────────────────────────────────────────
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

    # ── Purple test tile overlay ───────────────────────────────────────────────
    _test_trace_idx = None
    if not test_map_df.empty:
        _test_trace_idx = len(fig_map.data)
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

    # ── Click handler ─────────────────────────────────────────────────────────
    pts = (map_event.selection.points
           if map_event and map_event.selection else [])

    if pts:
        pt = pts[0]
        htext = pt.get("hovertext") or ""
        is_test = htext.startswith("TEST:")

        if is_test:
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
                        tp1, tp2 = generate_test_pngs_fn(clicked_file, clicked_mask, ft)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        tp1 = png_path_fn(clicked_file, "masks",   subdir=f"test_{model_stem}")
                        tp2 = png_path_fn(clicked_file, "overlay", subdir=f"test_{model_stem}")

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
                show_tile_metrics(ft)

        else:
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
                        p1, p2 = generate_pngs_fn(clicked_file, full_map)
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        p1 = png_path_fn(clicked_file, "masks")
                        p2 = png_path_fn(clicked_file, "overlay")

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
                show_tile_metrics(full_map)
