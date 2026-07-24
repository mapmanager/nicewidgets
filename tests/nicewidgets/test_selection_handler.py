"""Tests for PlotSelectionHandler rect/lasso selection and keyboard modifiers."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from nicewidgets.nicepool.dataframe_processor import DataFrameProcessor
from nicewidgets.nicepool.figure_generator import FigureGenerator
from nicewidgets.nicepool.plot_state import PlotState, PlotType
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE
from nicewidgets.nicepool.selection_handler import (
    PlotSelectionHandler,
    is_selection_compatible,
)


def _scatter_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pool_row_id": ["a", "b", "c"],
            "x": [1.0, 5.0, 9.0],
            "y": [1.0, 5.0, 9.0],
        }
    )


def _handler(df: pd.DataFrame | None = None) -> tuple[PlotSelectionHandler, list[set[str]], list[int]]:
    data = _scatter_df() if df is None else df
    processor = DataFrameProcessor(data, pre_filter_columns=[], unique_row_id_col="pool_row_id")
    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")
    applied: list[set[str]] = []
    label_counts: list[int] = []

    handler = PlotSelectionHandler(
        data_processor=processor,
        figure_generator=generator,
        unique_row_id_col="pool_row_id",
        get_filtered_df=lambda _plot_state: data,
        on_apply_selection=lambda: applied.append(handler.get_selected_row_ids()),
        on_update_label=label_counts.append,
    )
    return handler, applied, label_counts


@pytest.mark.parametrize(
    ("plot_type", "expected"),
    [
        (PlotType.SCATTER, True),
        (PlotType.SWARM, True),
        (PlotType.BOX_PLOT, False),
        (PlotType.VIOLIN, False),
        (PlotType.GROUPED, False),
        (PlotType.HISTOGRAM, False),
    ],
)
def test_is_selection_compatible_matches_documented_plot_types(
    plot_type: PlotType,
    expected: bool,
) -> None:
    """Only scatter and swarm plots should accept rect/lasso selection."""
    assert is_selection_compatible(plot_type) is expected


def test_handle_relayout_rect_selects_points_inside_range() -> None:
    """Rect selection payload should map to row ids in the chosen x/y window."""
    handler, applied, label_counts = _handler()
    state = PlotState(pre_filter={}, xcol="x", ycol="y", plot_type=PlotType.SCATTER)
    payload = {
        "selections": [{}],
        "selections[0].x0": 0.0,
        "selections[0].x1": 6.0,
        "selections[0].y0": 0.0,
        "selections[0].y1": 6.0,
    }

    handler.handle_relayout(payload, plot_index=0, plot_state=state)

    assert handler.get_selected_row_ids() == {"a", "b"}
    assert applied == [{"a", "b"}]
    assert label_counts == [2]


def test_handle_relayout_ignores_box_plot_selection_attempts() -> None:
    """Non-selection plot types must not mutate linked selection."""
    handler, applied, _ = _handler()
    state = PlotState(
        pre_filter={},
        xcol="x",
        ycol="y",
        plot_type=PlotType.BOX_PLOT,
        group_col="x",
    )
    handler.set_selected_row_ids({"seed"})
    payload = {
        "selections": [{}],
        "selections[0].x0": 0.0,
        "selections[0].x1": 10.0,
        "selections[0].y0": 0.0,
        "selections[0].y1": 10.0,
    }

    handler.handle_relayout(payload, plot_index=0, plot_state=state)

    assert handler.get_selected_row_ids() == {"seed"}
    assert applied == []


def test_handle_relayout_empty_selection_clears_existing_selection() -> None:
    """Clearing the Plotly selection box should clear linked row ids."""
    handler, applied, label_counts = _handler()
    state = PlotState(pre_filter={}, xcol="x", ycol="y", plot_type=PlotType.SCATTER)
    handler.set_selected_row_ids({"a"})

    handler.handle_relayout({"selections": []}, plot_index=0, plot_state=state)

    assert handler.get_selected_row_ids() == set()
    assert applied == [set()]
    assert label_counts == [0]


def test_handle_key_escape_clears_selection() -> None:
    """Escape should clear the current linked selection."""
    handler, applied, label_counts = _handler()
    handler.set_selected_row_ids({"a", "b"})

    handler.handle_key("Escape")

    assert handler.get_selected_row_ids() == set()
    assert applied == [set()]
    assert label_counts == [0]


def test_extend_modifier_unions_rect_selection_with_existing_ids() -> None:
    """Shift/meta extend should union new rect selection with the prior set."""
    handler, applied, _ = _handler()
    state = PlotState(pre_filter={}, xcol="x", ycol="y", plot_type=PlotType.SCATTER)
    handler.set_selected_row_ids({"a"})
    handler.handle_key("Meta", action=SimpleNamespace(keydown=True))

    handler.handle_relayout(
        {
            "selections": [{}],
            "selections[0].x0": 8.0,
            "selections[0].x1": 10.0,
            "selections[0].y0": 8.0,
            "selections[0].y1": 10.0,
        },
        plot_index=0,
        plot_state=state,
    )

    assert handler.get_selected_row_ids() == {"a", "c"}
    assert applied[-1] == {"a", "c"}


def test_select_by_row_id_matches_numeric_ids_when_passed_as_string() -> None:
    """Row-id lookup must compare string forms so table callbacks can pass str ids."""
    df = pd.DataFrame(
        {
            "pool_row_id": [1, 2],
            "parent": ["g1", "g1"],
            "velocity_mean": [1.0, 2.0],
        }
    )
    state = PlotState(
        pre_filter={"parent": PRE_FILTER_NONE},
        xcol="parent",
        ycol="velocity_mean",
        plot_type=PlotType.SWARM,
        group_col="parent",
    )
    processor = DataFrameProcessor(df, pre_filter_columns=("parent",), unique_row_id_col="pool_row_id")
    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")
    applied: list[set[str]] = []
    handler = PlotSelectionHandler(
        data_processor=processor,
        figure_generator=generator,
        unique_row_id_col="pool_row_id",
        get_filtered_df=lambda _plot_state: df,
        on_apply_selection=lambda: applied.append(handler.get_selected_row_ids()),
        on_update_label=lambda _count: None,
    )

    handler.select_by_row_id("1", [state])

    assert handler.get_selected_row_ids() == {"1"}
    assert applied == [{"1"}]


def test_select_by_row_id_noop_when_row_missing() -> None:
    """Missing row ids should leave selection unchanged without invoking apply callbacks."""
    handler, applied, _ = _handler()
    state = PlotState(pre_filter={}, xcol="x", ycol="y", plot_type=PlotType.SCATTER)
    handler.set_selected_row_ids({"a"})

    handler.select_by_row_id("missing", [state])

    assert handler.get_selected_row_ids() == {"a"}
    assert applied == []
