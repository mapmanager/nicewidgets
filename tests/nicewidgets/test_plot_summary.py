"""Tests for NicePool plot summary builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nicewidgets.nicepool.dataframe_processor import DataFrameProcessor
from nicewidgets.nicepool.figure_generator import FigureGenerator
from nicewidgets.nicepool.plot_state import PlotState, PlotType
from nicewidgets.nicepool.plot_summary import build_scatter_summary, build_swarm_summary
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE


def test_build_scatter_summary_when_xcol_equals_group_col() -> None:
    """Scatter summary must not duplicate the group column when xcol == group_col."""
    state = PlotState(
        pre_filter={"parent": PRE_FILTER_NONE},
        xcol="parent",
        ycol="velocity_mean",
        plot_type=PlotType.SCATTER,
        group_col="parent",
        color_grouping="roi_id",
    )
    tmp = pd.DataFrame(
        {
            "x": ["folder-a", "folder-a"],
            "y": [1.0, 2.0],
            "row_id": ["r1", "r2"],
            "file_stem": ["a", "a"],
            "color": ["folder-a", "folder-a"],
            "symbol": ["1", "2"],
        }
    )

    summary = build_scatter_summary(
        state,
        tmp,
        state.xcol,
        state.ycol,
        state.group_col,
        state.color_grouping,
    )

    assert "parent" in summary.columnar.columns
    assert "roi_id" in summary.columnar.columns
    assert list(summary.columnar.columns).count("parent") == 1


def test_build_swarm_summary_when_color_grouping_equals_group_col() -> None:
    """Swarm summary must not duplicate columns when color_grouping == group_col."""
    state = PlotState(
        pre_filter={"parent": PRE_FILTER_NONE},
        xcol="parent",
        ycol="velocity_mean",
        plot_type=PlotType.SWARM,
        group_col="parent",
        color_grouping="parent",
    )
    tmp = pd.DataFrame(
        {
            "x": ["folder-a", "folder-b"],
            "y": [1.0, 2.0],
            "row_id": ["r1", "r2"],
            "file_stem": ["a", "b"],
            "color": ["folder-a", "folder-b"],
        }
    )

    summary = build_swarm_summary(state, tmp, state.group_col, state.color_grouping)

    assert "parent" in summary.columnar.columns
    assert list(summary.columnar.columns).count("parent") == 1


def test_figure_generator_split_scatter_when_xcol_equals_group_col() -> None:
    """Split scatter replot must succeed when xcol and group_col share a name."""
    df = pd.DataFrame(
        [
            {
                "pool_row_id": "a",
                "parent": "folder-a",
                "roi_id": 1,
                "velocity_mean": 1.5,
            },
            {
                "pool_row_id": "b",
                "parent": "folder-a",
                "roi_id": 2,
                "velocity_mean": 2.5,
            },
        ]
    )
    state = PlotState(
        pre_filter={},
        xcol="parent",
        ycol="velocity_mean",
        plot_type=PlotType.SCATTER,
        group_col="parent",
        color_grouping="roi_id",
    )
    processor = DataFrameProcessor(df, pre_filter_columns=(), unique_row_id_col="pool_row_id")
    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")

    fig_dict, summary = generator.make_figure(df, state)

    assert fig_dict
    assert summary.columnar is not None
    assert "parent" in summary.columnar.columns


def test_stats_row_for_series_cv_is_nan_when_mean_near_zero() -> None:
    """CV in summary stats should guard near-zero means."""
    from nicewidgets.nicepool.plot_summary import stats_row_for_series

    y = pd.Series([-1.0, 1.0])
    row = stats_row_for_series(y, cv_epsilon=1e-10)

    assert np.isnan(row["cv"])
    assert row["count"] == 2
    assert row["mean"] == pytest.approx(0.0)


def test_build_histogram_summary_single_group() -> None:
    """Histogram summary should emit bin centers and counts for one trace."""
    from nicewidgets.nicepool.plot_summary import build_histogram_summary

    state = PlotState(
        pre_filter={},
        xcol="diameter",
        ycol="velocity_mean",
        plot_type=PlotType.HISTOGRAM,
        histogram_bins=4,
    )
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0])

    summary = build_histogram_summary(state, x, None, None, n_bins=4)

    assert summary.summary_table.iloc[0]["n"] == 5
    assert summary.summary_table.iloc[0]["count"] == 5
    assert len(summary.columnar) == 4
    assert set(summary.columnar.columns) == {"bin_center", "count"}


def test_build_histogram_summary_empty_input_has_documented_columns() -> None:
    """Empty histogram input should return empty tables with stable column names."""
    from nicewidgets.nicepool.plot_summary import build_histogram_summary

    state = PlotState(pre_filter={}, xcol="diameter", ycol="velocity_mean", plot_type=PlotType.HISTOGRAM)
    summary = build_histogram_summary(state, pd.Series(dtype=float), None, None)

    assert summary.summary_table.empty
    assert "n" in summary.summary_table.columns
    assert "count" in summary.summary_table.columns
    assert list(summary.columnar.columns) == ["bin_center", "count"]


def test_build_cumulative_histogram_summary_normalized_to_one() -> None:
    """Cumulative histogram summary should normalize the last bin to 1.0."""
    from nicewidgets.nicepool.plot_summary import build_cumulative_histogram_summary

    state = PlotState(
        pre_filter={},
        xcol="diameter",
        ycol="velocity_mean",
        plot_type=PlotType.CUMULATIVE_HISTOGRAM,
        histogram_bins=5,
    )
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    summary = build_cumulative_histogram_summary(state, x, None, None, None, None, n_bins=5)

    assert summary.summary_table.iloc[0]["n"] == 6
    assert summary.columnar["cumulative_proportion"].iloc[-1] == pytest.approx(1.0)


def test_build_cumulative_histogram_summary_with_color_grouping() -> None:
    """Grouped cumulative summary should include group and color columns."""
    from nicewidgets.nicepool.plot_summary import build_cumulative_histogram_summary

    state = PlotState(
        pre_filter={},
        xcol="diameter",
        ycol="velocity_mean",
        plot_type=PlotType.CUMULATIVE_HISTOGRAM,
        group_col="parent",
        color_grouping="roi_id",
        histogram_bins=3,
    )
    x = pd.Series([1.0, 2.0, 3.0, 4.0], index=[0, 1, 2, 3])
    group_series = pd.Series(["A", "A", "B", "B"], index=[0, 1, 2, 3])
    color_series = pd.Series(["1", "2", "1", "2"], index=[0, 1, 2, 3])

    summary = build_cumulative_histogram_summary(
        state,
        x,
        group_series,
        color_series,
        "parent",
        "roi_id",
        n_bins=3,
    )

    assert {"parent", "roi_id"}.issubset(summary.summary_table.columns)
    assert {"parent", "roi_id"}.issubset(summary.columnar.columns)
    assert len(summary.summary_table) == 4
