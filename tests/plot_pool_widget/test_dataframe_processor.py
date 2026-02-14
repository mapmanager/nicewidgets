"""Unit tests for DataFrameProcessor pre-filter API and filtering."""

import pytest
import pandas as pd

from nicewidgets.plot_pool_widget.dataframe_processor import DataFrameProcessor
from nicewidgets.plot_pool_widget.pre_filter_conventions import PRE_FILTER_NONE


@pytest.fixture
def sample_df():
    """DataFrame with pre_filter columns and unique_row_id_col."""
    return pd.DataFrame({
        "roi_id": [1, 1, 2, 2, 3],
        "condition": ["A", "B", "A", "B", "A"],
        "path": ["p1", "p2", "p3", "p4", "p5"],
        "value": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def processor(sample_df):
    return DataFrameProcessor(
        sample_df,
        pre_filter_columns=["roi_id", "condition"],
        unique_row_id_col="path",
    )


def test_get_pre_filter_values_returns_sorted_unique(processor):
    """get_pre_filter_values(column) returns sorted unique values for that column."""
    roi_vals = processor.get_pre_filter_values("roi_id")
    assert roi_vals == [1, 2, 3]
    cond_vals = processor.get_pre_filter_values("condition")
    assert cond_vals == ["A", "B"]


def test_get_pre_filter_values_unknown_column_raises(processor):
    """get_pre_filter_values(unknown_column) raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        processor.get_pre_filter_values("unknown")
    assert "Unknown pre_filter column" in str(exc_info.value)


def test_filter_by_pre_filters_all_none_returns_full_df(processor, sample_df):
    """filter_by_pre_filters with all (none) returns full df (minus dropna on row id)."""
    selections = {"roi_id": PRE_FILTER_NONE, "condition": PRE_FILTER_NONE}
    df_f = processor.filter_by_pre_filters(selections)
    assert len(df_f) == len(sample_df)
    pd.testing.assert_frame_equal(df_f.reset_index(drop=True), sample_df.reset_index(drop=True))


def test_filter_by_pre_filters_one_column_filters(processor):
    """filter_by_pre_filters with one column selected filters correctly."""
    selections = {"roi_id": 2, "condition": PRE_FILTER_NONE}
    df_f = processor.filter_by_pre_filters(selections)
    assert len(df_f) == 2
    assert set(df_f["roi_id"]) == {2}
    assert list(df_f["path"]) == ["p3", "p4"]


def test_filter_by_pre_filters_two_columns_and(processor):
    """filter_by_pre_filters with two columns applies AND."""
    selections = {"roi_id": 1, "condition": "A"}
    df_f = processor.filter_by_pre_filters(selections)
    assert len(df_f) == 1
    assert df_f.iloc[0]["path"] == "p1"


def test_filter_by_pre_filters_string_value_matches_numeric_column(processor):
    """UI sends string '2'; filter compares via str so it matches numeric 2."""
    selections = {"roi_id": "2", "condition": PRE_FILTER_NONE}
    df_f = processor.filter_by_pre_filters(selections)
    assert len(df_f) == 2
    assert set(df_f["roi_id"]) == {2}


def test_init_missing_pre_filter_column_raises(sample_df):
    """DataFrameProcessor with pre_filter_columns not in df raises."""
    with pytest.raises(ValueError) as exc_info:
        DataFrameProcessor(
            sample_df,
            pre_filter_columns=["roi_id", "missing_col"],
            unique_row_id_col="path",
        )
    assert "missing_col" in str(exc_info.value) or "pre_filter" in str(exc_info.value).lower()


def test_init_missing_unique_row_id_col_raises(sample_df):
    """DataFrameProcessor with unique_row_id_col not in df raises."""
    with pytest.raises(ValueError) as exc_info:
        DataFrameProcessor(
            sample_df,
            pre_filter_columns=["roi_id"],
            unique_row_id_col="missing",
        )
    assert "missing" in str(exc_info.value)
