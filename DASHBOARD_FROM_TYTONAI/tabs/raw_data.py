"""
Tab 6 — Raw Data

Parquet viewer for train + test metrics and a free-form DuckDB SQL console.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import duckdb
import pandas as pd
import streamlit as st


def render(
    con: duckdb.DuckDBPyConnection,
    model_stem: str,
    test_metrics_path_fn: Callable[[str], Path | None],
) -> None:

    def q(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    st.subheader("Raw data")

    st.markdown("**Training metrics — first 5 000 rows (sorted by F1 erosion ↑)**")
    st.dataframe(
        q("SELECT * FROM metrics ORDER BY f1_erosion ASC LIMIT 5000"),
        width="stretch", hide_index=True, height=420,
    )

    _tmet_raw = test_metrics_path_fn(model_stem)
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
