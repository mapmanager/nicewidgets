from __future__ import annotations

import pandas as pd

from nicewidgets.nicepool.dataframe_processor import DataFrameProcessor
from nicewidgets.nicepool.plot_preset_validation import sanitize_layout, sanitize_preset_payload, sanitize_plot_state
from nicewidgets.nicepool.plot_state import PlotState, PlotType
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pool_row_id": ["a", "b", "c"],
            "accept": [True, False, True],
            "channel": [1, 1, 2],
            "roi_id": ["r1", "r2", "r1"],
            "parent": ["cond1", "cond2", "cond1"],
            "velocity_mean": [1.0, 2.0, 3.0],
            "heart_rate": [10.0, 11.0, 12.0],
        }
    )


def _processor(df: pd.DataFrame) -> DataFrameProcessor:
    return DataFrameProcessor(
        df,
        pre_filter_columns=["accept", "channel", "roi_id"],
        unique_row_id_col="pool_row_id",
    )


def _default_state() -> PlotState:
    return PlotState(
        pre_filter={"accept": PRE_FILTER_NONE, "channel": PRE_FILTER_NONE, "roi_id": PRE_FILTER_NONE},
        xcol="parent",
        ycol="velocity_mean",
        plot_type=PlotType.SWARM,
        group_col="parent",
        color_grouping="roi_id",
    )


def test_sanitize_layout_falls_back_for_unknown_layout():
    assert sanitize_layout("bogus") == "1x1"
    assert sanitize_layout("2x2") == "2x2"


def test_sanitize_plot_state_repairs_stale_columns_and_prefilter_values():
    df = _df()
    stale = PlotState(
        pre_filter={"accept": "missing", "channel": "1", "roi_id": "missing", "old": "x"},
        xcol="old_x",
        ycol="old_y",
        plot_type=PlotType.SCATTER,
        group_col="old_group",
        color_grouping="old_color",
        ystat="not-a-stat",
        std_sem_type="bad",
    )

    state = sanitize_plot_state(
        stale,
        df=df,
        data_processor=_processor(df),
        pre_filter_columns=["accept", "channel", "roi_id"],
        default_state=_default_state(),
    )

    assert state.xcol == "parent"
    assert state.ycol == "velocity_mean"
    assert state.group_col is None
    assert state.color_grouping is None
    assert state.pre_filter == {"accept": PRE_FILTER_NONE, "channel": "1", "roi_id": PRE_FILTER_NONE}
    assert state.ystat == "mean"
    assert state.std_sem_type == "std"


def test_sanitize_preset_payload_ignores_unknown_keys_and_pads_states():
    df = _df()
    payload = {
        "layout": "1x2",
        "unknown": "ignored",
        "plot_states": [
            {
                "pre_filter": {"accept": "True", "channel": "2", "roi_id": "r1"},
                "xcol": "parent",
                "ycol": "heart_rate",
                "plot_type": "swarm",
                "group_col": "parent",
                "color_grouping": "roi_id",
            }
        ],
    }

    layout, states = sanitize_preset_payload(
        payload,
        df=df,
        data_processor=_processor(df),
        pre_filter_columns=["accept", "channel", "roi_id"],
        default_state=_default_state(),
    )

    assert layout == "1x2"
    assert len(states) == 2
    assert states[0].ycol == "heart_rate"
    assert states[0].pre_filter == {"accept": "True", "channel": "2", "roi_id": "r1"}
    assert states[1].ycol == "velocity_mean"
