"""
Streamlit-cached resources: DuckDB connections, PyTorch model, tile map.

Exports
-------
get_con, get_test_con, get_model, tile_map
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import streamlit as st
import torch


@st.cache_resource
def get_con(path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    p = Path(path)
    if p.suffix == ".parquet":
        con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_parquet('{p}')")
    else:
        con.execute(f"CREATE VIEW metrics AS SELECT * FROM read_csv_auto('{p}')")
    return con


@st.cache_resource
def get_test_con(path: str) -> duckdb.DuckDBPyConnection:
    _c = duckdb.connect()
    _c.execute(f"CREATE VIEW test_metrics AS SELECT * FROM read_parquet('{path}')")
    return _c


@st.cache_resource(show_spinner="Loading model on CPU…")
def get_model(model_path: str):
    from src.model import load_model
    device = torch.device("cpu")
    model = load_model(Path(model_path), device=device)
    return model, device


@st.cache_data(show_spinner=False)
def tile_map(model_name: str) -> dict:
    from src.dataset import load_tiles_json
    from services.registry import _model_tiles_json
    return {t["imagery_file"]: t for t in load_tiles_json(_model_tiles_json(model_name))}
