"""Unit tests for PlotState serialization and pre_filter (schema v3)."""

import pytest

from nicewidgets.plot_pool_widget.plot_state import PlotState, PlotType
from nicewidgets.plot_pool_widget.pre_filter_conventions import PRE_FILTER_NONE


def test_plot_state_to_dict_contains_pre_filter():
    """to_dict() must include pre_filter (no roi_id)."""
    state = PlotState(
        pre_filter={"roi_id": PRE_FILTER_NONE, "condition": "A"},
        xcol="x",
        ycol="y",
    )
    d = state.to_dict()
    assert "pre_filter" in d
    assert d["pre_filter"] == {"roi_id": PRE_FILTER_NONE, "condition": "A"}
    assert "roi_id" not in d


def test_plot_state_from_dict_round_trip():
    """from_dict(to_dict(state)) equals state for pre_filter and key fields."""
    state = PlotState(
        pre_filter={"roi_id": "1"},
        xcol="img_mean",
        ycol="vel_mean",
        plot_type=PlotType.VIOLIN,
        group_col="group",
    )
    d = state.to_dict()
    restored = PlotState.from_dict(d)
    assert restored.pre_filter == state.pre_filter
    assert restored.xcol == state.xcol
    assert restored.ycol == state.ycol
    assert restored.plot_type == state.plot_type
    assert restored.group_col == state.group_col


def test_plot_state_from_dict_rejects_roi_id():
    """from_dict() raises ValueError if data contains legacy roi_id."""
    data = {
        "roi_id": 1,
        "xcol": "x",
        "ycol": "y",
        "pre_filter": {"roi_id": "1"},
    }
    with pytest.raises(ValueError) as exc_info:
        PlotState.from_dict(data)
    assert "roi_id" in str(exc_info.value)
    assert "pre_filter" in str(exc_info.value)


def test_plot_state_from_dict_accepts_pre_filter_only():
    """from_dict() with pre_filter and no roi_id succeeds."""
    data = {
        "pre_filter": {"roi_id": "(none)"},
        "xcol": "a",
        "ycol": "b",
    }
    state = PlotState.from_dict(data)
    assert state.pre_filter == {"roi_id": "(none)"}
    assert state.xcol == "a"
    assert state.ycol == "b"


def test_plot_state_from_dict_missing_pre_filter_defaults_empty():
    """from_dict() with missing pre_filter uses empty dict."""
    data = {"xcol": "a", "ycol": "b"}
    state = PlotState.from_dict(data)
    assert state.pre_filter == {}
