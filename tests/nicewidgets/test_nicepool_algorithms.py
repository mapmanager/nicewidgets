"""Pure-Python tests for NicePool reference algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from nicewidgets.nicepool.algorithms.group_plot import (
    PRE_FILTER_NONE,
    filter_by_pre_filters as filter_group_pre_filters,
    group_plot_algorithm,
    grouped_full_stats_table,
)
from nicewidgets.nicepool.algorithms.intv_stats import (
    compute_iei_and_inst_freq,
    filter_zero_iei,
    intv_stats,
    parse_rel_path,
)
from nicewidgets.nicepool.algorithms.swarm_stats import (
    histogram_values_per_group,
    prepare_scatter_tmp,
    prepare_swarm_tmp,
    scatter_full_stats_table,
    scatter_values_per_group,
    swarm_full_stats_table,
    swarm_values_per_group,
)


def _pool_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pool_row_id": ["r0", "r1", "r2", "r3", None, "r5"],
            "roi_id": ["1", "1", "2", "2", "1", "1"],
            "condition": ["A", "A", "A", "B", "B", "B"],
            "channel": [1, 1, 1, 2, 2, 2],
            "velocity": [1.0, -2.0, 3.0, 4.0, 5.0, 100.0],
            "diameter": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        }
    )


def test_group_plot_algorithm_filters_rows_and_aggregates_mean() -> None:
    """Grouped algorithm should match the documented filter + aggregate pipeline."""
    result = group_plot_algorithm(
        _pool_df(),
        pre_filter_columns=["roi_id"],
        unique_row_id_col="pool_row_id",
        pre_filter_selections={"roi_id": "1"},
        group_col="condition",
        ycol="velocity",
        ystat="mean",
        use_remove_values=True,
        remove_values_threshold=10.0,
    )

    expected = pd.Series({"A": -0.5, "B": np.nan}, name="y")
    expected.index.name = "group"
    pdt.assert_series_equal(result, expected)


def test_grouped_full_stats_table_computes_cv_with_zero_mean_guard() -> None:
    """CV should be NaN when group mean is smaller than cv_epsilon."""
    df = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "velocity": [-1.0, 1.0, 2.0, 4.0],
        }
    )

    stats = grouped_full_stats_table(df, group_col="condition", ycol="velocity", cv_epsilon=1e-10)

    assert np.isnan(stats.loc["A", "cv"])
    assert stats.loc["B", "mean"] == 3.0
    assert stats.loc["B", "count"] == 2


def test_group_filter_drops_missing_unique_row_ids_after_prefilter() -> None:
    """Reference filter should remove rows without the configured row-id column."""
    filtered = filter_group_pre_filters(
        _pool_df(),
        pre_filter_columns=["roi_id"],
        selections={"roi_id": PRE_FILTER_NONE},
        unique_row_id_col="pool_row_id",
    )

    assert filtered["pool_row_id"].isna().sum() == 0
    assert len(filtered) == 5


def test_intv_stats_filters_zero_intervals_before_aggregation() -> None:
    """Zero IEI values are preserved as NaN and excluded from summary stats."""
    ts = pd.Series([0.0, 0.5, 0.5, 1.5])
    iei, inst_freq = compute_iei_and_inst_freq(ts)
    iei_f, inst_freq_f, n_original = filter_zero_iei(iei, inst_freq)

    assert n_original == 3
    assert iei_f.iloc[1] == 0.5
    assert np.isnan(iei_f.iloc[2])
    assert iei_f.iloc[3] == 1.0
    assert inst_freq_f.iloc[1] == 2.0
    assert np.isnan(inst_freq_f.iloc[2])
    assert inst_freq_f.iloc[3] == 1.0


def test_intv_stats_adds_path_context_columns() -> None:
    """Interval stats should parse rel_path into context columns in the result table."""
    rel_path = "condA/session1/file.tif"
    df = pd.DataFrame(
        {
            "roi_id": ["1", "1", "1"],
            "rel_path": [rel_path, rel_path, rel_path],
            "event_type": ["peak", "peak", "peak"],
            "t_start": [0.0, 0.25, 0.75],
        }
    )

    result = intv_stats(df, time_col="t_start", roi_id="1", rel_path=rel_path, event_type="peak")
    table = result["table"]

    assert parse_rel_path(rel_path) == {"grandparent": "condA", "parent": "session1", "tif_file": "file.tif"}
    assert table.loc["iei", "grandparent"] == "condA"
    assert table.loc["iei", "parent"] == "session1"
    assert table.loc["iei", "tif_file"] == "file.tif"
    assert table.loc["iei", "count"] == 2


def test_swarm_stats_table_and_values_support_color_grouping() -> None:
    """Swarm helpers should group by both x and color when color grouping exists."""
    tmp = prepare_swarm_tmp(
        _pool_df().dropna(subset=["pool_row_id"]),
        group_col="condition",
        ycol="velocity",
        color_grouping="roi_id",
        use_absolute=True,
        use_remove_values=True,
        remove_values_threshold=10.0,
    )

    stats = swarm_full_stats_table(tmp, group_col="condition", color_grouping="roi_id")
    values = swarm_values_per_group(tmp)

    assert list(stats[["condition", "roi_id"]].itertuples(index=False, name=None)) == [
        ("A", "1"),
        ("A", "2"),
        ("B", "2"),
    ]
    assert values == {"A_1": [1.0, 2.0], "A_2": [3.0], "B_2": [4.0]}


def test_scatter_stats_and_values_use_explicit_groups() -> None:
    """Scatter helpers should emit x/y lists for each explicit group."""
    tmp = prepare_scatter_tmp(
        _pool_df().dropna(subset=["pool_row_id"]),
        xcol="diameter",
        ycol="velocity",
        group_col="condition",
        use_remove_values=True,
        remove_values_threshold=20.0,
    )

    stats = scatter_full_stats_table(tmp)
    values = scatter_values_per_group(tmp)

    assert list(stats["group"]) == ["A", "B"]
    assert stats.loc[stats["group"] == "A", "count"].iloc[0] == 3
    assert stats.loc[stats["group"] == "B", "count"].iloc[0] == 1
    assert values == {
        "x_A": [10.0, 11.0, 12.0],
        "y_A": [1.0, -2.0, 3.0],
        "x_B": [13.0],
        "y_B": [4.0],
    }


def test_histogram_values_per_group_returns_counts_by_group() -> None:
    """Histogram helper should return bin centers and counts per group."""
    values = histogram_values_per_group(
        _pool_df().dropna(subset=["pool_row_id"]),
        xcol="diameter",
        group_col="condition",
        nbins=2,
    )

    assert set(values) == {"x_A", "y_A", "x_B", "y_B"}
    assert values["y_A"] == [1, 2]
    assert values["y_B"] == [1, 1]
