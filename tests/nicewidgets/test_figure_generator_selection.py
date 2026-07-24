"""Tests for NicePool selection overlay rendering."""

from __future__ import annotations

import base64

import numpy as np
import pandas as pd

from nicewidgets.nicepool.dataframe_processor import DataFrameProcessor
from nicewidgets.nicepool.figure_generator import (
    SELECTION_OVERLAY_TRACE_NAME,
    SELECTED_POINTS_SIZE_MULTIPLIER,
    FigureGenerator,
)
from nicewidgets.nicepool.plot_state import PlotState, PlotType
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pool_row_id": "a",
                "parent": "folder-a",
                "roi_id": 1,
                "accept": True,
                "channel": 0,
                "velocity_mean": 1.5,
                "path": "/data/f1.oir",
            },
            {
                "pool_row_id": "b",
                "parent": "folder-a",
                "roi_id": 2,
                "accept": True,
                "channel": 0,
                "velocity_mean": 2.5,
                "path": "/data/f2.oir",
            },
            {
                "pool_row_id": "c",
                "parent": "folder-b",
                "roi_id": 1,
                "accept": True,
                "channel": 1,
                "velocity_mean": 3.0,
                "path": "/data/f3.oir",
            },
        ]
    )


def _cloudscope_plot_state(plot_type: PlotType) -> PlotState:
    return PlotState(
        pre_filter={
            "accept": PRE_FILTER_NONE,
            "channel": PRE_FILTER_NONE,
            "roi_id": PRE_FILTER_NONE,
        },
        xcol="parent",
        ycol="velocity_mean",
        plot_type=plot_type,
        group_col="parent",
        color_grouping="roi_id",
        point_size=6,
    )


def _overlay_traces(fig_dict: dict) -> list[dict]:
    return [trace for trace in fig_dict["data"] if trace.get("name") == SELECTION_OVERLAY_TRACE_NAME]


def _overlay_point_count(overlay_trace: dict) -> int:
    """Return the number of points in a Plotly scatter trace dict."""
    x = overlay_trace["x"]
    if isinstance(x, dict):
        return len(np.frombuffer(base64.b64decode(x["bdata"]), dtype=x["dtype"]))
    return len(x)


def test_scatter_selection_uses_overlay_trace() -> None:
    """Split scatter should render selection via overlay, not selectedpoints."""
    df = _sample_df()
    state = _cloudscope_plot_state(PlotType.SCATTER)
    processor = DataFrameProcessor(
        df,
        pre_filter_columns=("accept", "channel", "roi_id"),
        unique_row_id_col="pool_row_id",
    )
    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")
    df_f = processor.filter_by_pre_filters(state.pre_filter)

    fig, _ = generator.make_figure(df_f, state, selected_row_ids={"b"})

    overlays = _overlay_traces(fig)
    assert len(overlays) == 1
    assert overlays[0]["showlegend"] is False
    assert overlays[0]["marker"]["size"] == state.point_size * SELECTED_POINTS_SIZE_MULTIPLIER
    assert overlays[0]["marker"]["symbol"] == "circle"
    assert _overlay_point_count(overlays[0]) == 1
    assert all(trace.get("selectedpoints") is None for trace in fig["data"] if trace is not overlays[0])


def test_swarm_selection_uses_overlay_trace() -> None:
    """Swarm should also use the shared overlay trace for consistent highlighting."""
    df = _sample_df()
    state = _cloudscope_plot_state(PlotType.SWARM)
    processor = DataFrameProcessor(
        df,
        pre_filter_columns=("accept", "channel", "roi_id"),
        unique_row_id_col="pool_row_id",
    )
    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")
    df_f = processor.filter_by_pre_filters(state.pre_filter)

    fig, _ = generator.make_figure(df_f, state, selected_row_ids={"b"})

    overlays = _overlay_traces(fig)
    assert len(overlays) == 1
    assert _overlay_point_count(overlays[0]) == 1


def test_no_overlay_when_selection_empty() -> None:
    df = _sample_df()
    state = _cloudscope_plot_state(PlotType.SCATTER)
    processor = DataFrameProcessor(
        df,
        pre_filter_columns=("accept", "channel", "roi_id"),
        unique_row_id_col="pool_row_id",
    )
    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")
    df_f = processor.filter_by_pre_filters(state.pre_filter)

    fig, _ = generator.make_figure(df_f, state, selected_row_ids=set())

    assert _overlay_traces(fig) == []
