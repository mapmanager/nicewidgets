"""Tests for categorical validation used by NicePool figure generation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nicewidgets.nicepool.plot_errors import PlotDataError, prepare_categorical_column
from nicewidgets.nicepool.plot_pool_controller import PlotPoolConfig, PlotPoolController
from nicewidgets.nicepool.plot_state import PlotState, PlotType


def test_prepare_categorical_column_raises_on_mixed_int_and_str() -> None:
    """Mixed numeric/text metadata values should fail with a clear plot error."""
    series = pd.Series([2, "2", 3])

    with pytest.raises(PlotDataError, match="mixed value types"):
        prepare_categorical_column(series, column_name="branch_order", role="group axis")


def test_prepare_categorical_column_allows_mixed_int_and_float() -> None:
    """Integer and float buckets may represent the same numeric category axis."""
    labels, unique = prepare_categorical_column(
        pd.Series([2, 2.0, 3]),
        column_name="branch_order",
        role="group axis",
    )

    assert unique == ["2.0", "3.0"]
    assert labels.dropna().tolist() == ["2.0", "2.0", "3.0"]


def test_prepare_categorical_column_raises_when_all_missing() -> None:
    """All-missing categorical columns should fail before sorting."""
    with pytest.raises(PlotDataError, match="No valid values"):
        prepare_categorical_column(
            pd.Series([pd.NA, None, pd.NA]),
            column_name="branch_order",
            role="group axis",
        )


def test_swarm_plot_with_color_grouping_does_not_crash_on_nullable_int_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swarm color grouping on nullable Int64 roi_id should not fail sorting colors."""
    df = pd.DataFrame(
        {
            "pool_row_id": ["r0", "r1", "r2"],
            "parent": ["g1", "g1", "g2"],
            "branch_order": pd.array([2, 2, 3], dtype="Int64"),
            "roi_id": pd.array([1, 2, 1], dtype="Int64"),
            "velocity_mean": [1.0, 2.0, 3.0],
        }
    )
    controller = PlotPoolController(
        df,
        config=PlotPoolConfig(
            unique_row_id_col="pool_row_id",
            pre_filter_columns=[],
            enable_config_persistence=False,
            plot_state=PlotState(
                pre_filter={},
                xcol="parent",
                ycol="velocity_mean",
                plot_type=PlotType.SWARM,
                group_col="branch_order",
                color_grouping="roi_id",
            ),
        ),
    )
    monkeypatch.setattr(
        "nicewidgets.nicepool.plot_pool_controller.ui.notify",
        MagicMock(),
    )
    monkeypatch.setattr(
        "nicewidgets.nicepool.plot_pool_controller.ui.plotly",
        MagicMock(from_dict=lambda *_args, **_kwargs: MagicMock(update=MagicMock())),
    )

    figure_dict = controller._make_figure_dict(controller.plot_states[0])

    assert figure_dict["data"]


def test_swarm_plot_notifies_on_mixed_branch_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swarm group/color on mixed branch_order should notify instead of crashing."""
    df = pd.DataFrame(
        {
            "pool_row_id": ["r0", "r1", "r2"],
            "branch_order": [2, "2", pd.NA],
            "velocity_mean": [1.0, 2.0, 3.0],
        }
    )
    controller = PlotPoolController(
        df,
        config=PlotPoolConfig(
            unique_row_id_col="pool_row_id",
            pre_filter_columns=[],
            enable_config_persistence=False,
            plot_state=PlotState(
                pre_filter={},
                xcol="velocity_mean",
                ycol="velocity_mean",
                plot_type=PlotType.SWARM,
                group_col="branch_order",
            ),
        ),
    )
    notifications: list[tuple[str, str]] = []
    logged: list[str] = []
    monkeypatch.setattr(
        "nicewidgets.nicepool.plot_pool_controller.ui.notify",
        lambda message, *, type="info": notifications.append((message, type)),
    )
    monkeypatch.setattr(
        "nicewidgets.nicepool.plot_pool_controller.logger.warning",
        lambda message, *args: logged.append(message % args if args else message),
    )
    monkeypatch.setattr(
        "nicewidgets.nicepool.plot_pool_controller.ui.plotly",
        MagicMock(from_dict=lambda *_args, **_kwargs: MagicMock(update=MagicMock())),
    )

    figure_dict = controller._make_figure_dict(controller.plot_states[0])

    assert figure_dict["data"] == []
    annotation_text = figure_dict["layout"]["annotations"][0]["text"].replace("<br>", " ")
    assert "mixed value types" in annotation_text.lower()
    assert notifications
    assert notifications[0][1] == "warning"
    assert logged
    assert "mixed value types" in logged[0].lower()
