"""
Shared UI helpers reused across multiple tabs.

Exports
-------
fmt, show_tile_metrics
"""
from __future__ import annotations

import pandas as pd
import streamlit as st


def fmt(v: object, dec: int = 4) -> str:
    """Format a float value safely, returning 'N/A' for NaN."""
    try:
        f = float(v)
        return "N/A" if (f != f) else f"{f:.{dec}f}"
    except (TypeError, ValueError):
        return "N/A"


def show_tile_metrics(full: dict) -> None:
    """Render a two-column Erosion / No-erosion metrics table."""
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
