"""Tests for FigureGenerator plot-type routing, transforms, and reference parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from nicewidgets.nicepool.algorithms.group_plot import group_plot_algorithm, grouped_aggregate
from nicewidgets.nicepool.dataframe_processor import DataFrameProcessor
from nicewidgets.nicepool.figure_generator import FigureGenerator
from nicewidgets.nicepool.plot_state import PlotState, PlotType
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE


def _pool_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pool_row_id": ["a", "b", "c", "d", "e"],
            "parent": ["folder-a", "folder-a", "folder-b", "folder-b", "folder-b"],
            "roi_id": [1, 2, 1, 2, 3],
            "channel": [0, 0, 1, 1, 1],
            "velocity_mean": [1.5, -2.5, 3.0, 4.0, 100.0],
            "diameter": [10.0, 11.0, 12.0, 13.0, 14.0],
            "path": ["/data/f1.oir", "/data/f2.oir", "/data/f3.oir", "/data/f4.oir", "/data/f5.oir"],
        }
    )


def _many_row_df(n: int = 30) -> pd.DataFrame:
    """DataFrame whose numeric x-index column is not a categorical candidate."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 100, size=n)
    return pd.DataFrame(
        {
            "pool_row_id": [f"r{i}" for i in range(n)],
            "parent": ["g1"] * (n // 2) + ["g2"] * (n - n // 2),
            "channel": [0] * n,
            "roi_id": list(range(n)),
            "velocity_mean": xs,
            "diameter": rng.uniform(5, 20, size=n),
            "path": [f"/data/f{i}.oir" for i in range(n)],
        }
    )


def _processor(df: pd.DataFrame) -> DataFrameProcessor:
    return DataFrameProcessor(
        df,
        pre_filter_columns=["channel", "roi_id"],
        unique_row_id_col="pool_row_id",
    )


def _generator(df: pd.DataFrame) -> FigureGenerator:
    processor = _processor(df)
    return FigureGenerator(processor, unique_row_id_col="pool_row_id")


def _make(
    df: pd.DataFrame,
    state: PlotState,
    *,
    selected_row_ids: set[str] | None = None,
) -> tuple[dict, object]:
    generator = _generator(df)
    df_f = _processor(df).filter_by_pre_filters(state.pre_filter)
    return generator.make_figure(df_f, state, selected_row_ids=selected_row_ids)


def _base_state(**overrides: object) -> PlotState:
    state = PlotState(
        pre_filter={"channel": PRE_FILTER_NONE, "roi_id": PRE_FILTER_NONE},
        xcol="diameter",
        ycol="velocity_mean",
        plot_type=PlotType.SCATTER,
        group_col="parent",
        color_grouping="roi_id",
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


@pytest.mark.parametrize(
    "plot_type",
    [
        PlotType.GROUPED,
        PlotType.BOX_PLOT,
        PlotType.VIOLIN,
        PlotType.SWARM,
        PlotType.HISTOGRAM,
        PlotType.CUMULATIVE_HISTOGRAM,
    ],
)
def test_make_figure_produces_traces_for_each_plot_type(plot_type: PlotType) -> None:
    """Each supported plot type should return a figure and summary without raising."""
    df = _pool_df()
    state = _base_state(plot_type=plot_type)
    if plot_type == PlotType.HISTOGRAM:
        state = _base_state(plot_type=plot_type, group_col=None)
    if plot_type == PlotType.GROUPED:
        state = _base_state(plot_type=plot_type, ystat="mean")

    fig, summary = _make(df, state)

    assert fig["data"]
    assert summary.params["plot_type"] == plot_type.value
    assert summary.columnar is not None


def test_box_plot_raises_when_group_col_is_high_cardinality_numeric() -> None:
    """Non-categorical group_col must fail fast instead of silently changing plot type."""
    from nicewidgets.nicepool.plot_errors import PlotConfigurationError

    df = _many_row_df()
    state = _base_state(
        plot_type=PlotType.BOX_PLOT,
        group_col="velocity_mean",
        color_grouping=None,
    )

    with pytest.raises(PlotConfigurationError, match="categorical"):
        _make(df, state)


def test_histogram_raises_when_x_column_has_no_valid_values_after_filter() -> None:
    """Histogram must fail fast when filtered data has no numeric x values."""
    from nicewidgets.nicepool.plot_errors import PlotDataError

    df = _pool_df()
    state = _base_state(
        plot_type=PlotType.HISTOGRAM,
        group_col=None,
        xcol="diameter",
        pre_filter={"channel": "999", "roi_id": PRE_FILTER_NONE},
    )

    with pytest.raises(PlotDataError, match="no valid values"):
        _make(df, state)


def test_grouped_plot_raises_when_group_col_missing() -> None:
    """Grouped plots must not silently switch to another plot type."""
    from nicewidgets.nicepool.plot_errors import PlotConfigurationError

    df = _pool_df()
    state = _base_state(plot_type=PlotType.GROUPED, ystat="mean", group_col=None)

    with pytest.raises(PlotConfigurationError, match="Group column"):
        _make(df, state)


def test_swarm_plot_uses_box_trace_when_group_col_is_categorical() -> None:
    """Categorical parent column should produce a swarm/box-style trace, not scatter fallback."""
    df = _pool_df()
    state = _base_state(plot_type=PlotType.SWARM)

    fig, summary = _make(df, state)

    assert summary.params["plot_type"] == PlotType.SWARM.value
    assert fig["data"]


def test_grouped_plot_matches_reference_algorithm_for_mean() -> None:
    """FigureGenerator grouped output must match the documented reference pipeline."""
    df = _pool_df()
    state = _base_state(
        plot_type=PlotType.GROUPED,
        ystat="mean",
        use_remove_values=True,
        remove_values_threshold=50.0,
    )
    df_f = _processor(df).filter_by_pre_filters(state.pre_filter)

    expected = group_plot_algorithm(
        df,
        pre_filter_columns=["channel", "roi_id"],
        unique_row_id_col="pool_row_id",
        pre_filter_selections=state.pre_filter,
        group_col=state.group_col,
        ycol=state.ycol,
        ystat=state.ystat,
        use_absolute_value=state.use_absolute_value,
        use_remove_values=state.use_remove_values,
        remove_values_threshold=state.remove_values_threshold,
        cv_epsilon=state.cv_epsilon,
    )

    fig, _ = _generator(df).make_figure(df_f, state)
    trace = fig["data"][0]

    actual = pd.Series(trace["y"], index=trace["x"], name="y")
    actual.index.name = "group"
    pdt.assert_series_equal(actual.sort_index(), expected.sort_index())


def test_grouped_cv_stat_respects_cv_epsilon() -> None:
    """CV aggregation should emit NaN when |mean| is below PlotState.cv_epsilon."""
    df = pd.DataFrame(
        {
            "pool_row_id": ["a", "b", "c", "d"],
            "parent": ["A", "A", "B", "B"],
            "velocity_mean": [-1.0, 1.0, 2.0, 4.0],
        }
    )
    state = PlotState(
        pre_filter={},
        xcol="velocity_mean",
        ycol="velocity_mean",
        plot_type=PlotType.GROUPED,
        group_col="parent",
        ystat="cv",
        cv_epsilon=1e-10,
    )
    processor = DataFrameProcessor(df, pre_filter_columns=[], unique_row_id_col="pool_row_id")
    df_f = processor.filter_by_pre_filters(state.pre_filter)

    expected = grouped_aggregate(
        df_f,
        group_col="parent",
        ycol="velocity_mean",
        ystat="cv",
        cv_epsilon=state.cv_epsilon,
    )

    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")
    fig, _ = generator.make_figure(df_f, state)
    actual = pd.Series(fig["data"][0]["y"], index=fig["data"][0]["x"])

    assert np.isnan(expected.loc["A"])
    assert np.isnan(actual.loc["A"])
    assert expected.loc["B"] == pytest.approx(actual.loc["B"])


def test_use_absolute_value_transforms_negative_y_before_plotting() -> None:
    """Absolute-value flag should affect grouped aggregation results."""
    df = _pool_df()
    state = _base_state(
        plot_type=PlotType.GROUPED,
        ystat="mean",
        use_absolute_value=True,
    )
    df_f = _processor(df).filter_by_pre_filters(state.pre_filter)

    fig, _ = _generator(df).make_figure(df_f, state)
    folder_a_y = fig["data"][0]["y"][fig["data"][0]["x"].index("folder-a")]

    expected = grouped_aggregate(
        df_f,
        group_col="parent",
        ycol="velocity_mean",
        ystat="mean",
        use_absolute=True,
    )
    assert folder_a_y == pytest.approx(float(expected.loc["folder-a"]))


def test_empty_filtered_dataframe_does_not_raise_for_grouped_plot() -> None:
    """Empty filtered input should degrade gracefully."""
    df = _pool_df()
    state = _base_state(
        plot_type=PlotType.GROUPED,
        ystat="mean",
        pre_filter={"channel": "999", "roi_id": PRE_FILTER_NONE},
    )

    fig, summary = _make(df, state)

    assert fig["data"]
    assert summary.summary_table is not None


def test_histogram_summary_columnar_matches_direct_builder() -> None:
    """Histogram figure path should stay consistent with build_histogram_summary."""
    from nicewidgets.nicepool.plot_summary import build_histogram_summary

    df = _pool_df()
    state = _base_state(plot_type=PlotType.HISTOGRAM, group_col=None, xcol="diameter")

    fig, summary = _make(df, state)
    x = _processor(df).get_x_values(
        _processor(df).filter_by_pre_filters(state.pre_filter),
        state.xcol,
        state.use_absolute_value,
        state.use_remove_values,
        state.remove_values_threshold,
    ).dropna()
    direct = build_histogram_summary(state, x, None, state.group_col, n_bins=state.histogram_bins)

    assert summary.columnar.shape == direct.columnar.shape
    assert summary.summary_table.iloc[0]["n"] == direct.summary_table.iloc[0]["n"]


def _trace_y_values(trace: dict) -> list[float]:
    """Return y values from a Plotly trace dict (plain list or bdata encoding)."""
    y = trace["y"]
    if isinstance(y, dict):
        import base64

        arr = np.frombuffer(base64.b64decode(y["bdata"]), dtype=y["dtype"])
        return arr.tolist()
    return list(y)


def test_cumulative_histogram_endpoints_are_monotonic() -> None:
    """Cumulative histogram traces should start at zero and end near one."""
    df = _pool_df()
    state = _base_state(
        plot_type=PlotType.CUMULATIVE_HISTOGRAM,
        group_col=None,
        xcol="diameter",
    )

    fig, summary = _make(df, state)
    y = _trace_y_values(fig["data"][0])

    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(1.0)
    assert summary.columnar["cumulative_proportion"].iloc[-1] == pytest.approx(1.0)
