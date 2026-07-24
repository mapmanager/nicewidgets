"""Tests for NicePool plot error handling in the controller layer."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from nicewidgets.nicepool.plot_pool_controller import PlotPoolConfig, PlotPoolController
from nicewidgets.nicepool.plot_state import PlotState, PlotType


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pool_row_id": [f"r{i}" for i in range(30)],
            "parent": ["g1"] * 15 + ["g2"] * 15,
            "velocity_mean": [float(i) for i in range(30)],
            "diameter": [float(10 + i) for i in range(30)],
        }
    )


def test_make_figure_dict_notifies_and_returns_empty_figure_on_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Controller should surface configuration failures without changing plot type."""
    controller = PlotPoolController(
        _df(),
        config=PlotPoolConfig(
            unique_row_id_col="pool_row_id",
            pre_filter_columns=[],
            enable_config_persistence=False,
        ),
    )
    notifications: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "nicewidgets.nicepool.plot_pool_controller.ui.notify",
        lambda message, *, type="info": notifications.append((message, type)),
    )

    state = PlotState(
        pre_filter={},
        xcol="diameter",
        ycol="velocity_mean",
        plot_type=PlotType.BOX_PLOT,
        group_col="velocity_mean",
    )

    figure_dict = controller._make_figure_dict(state)

    assert figure_dict["data"] == []
    assert "categorical" in figure_dict["layout"]["annotations"][0]["text"].lower()
    assert notifications
    assert notifications[0][1] == "warning"


def test_make_figure_dict_returns_valid_figure_for_supported_swarm_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid swarm configuration should still render normally."""
    controller = PlotPoolController(
        _df(),
        config=PlotPoolConfig(
            unique_row_id_col="pool_row_id",
            pre_filter_columns=[],
            enable_config_persistence=False,
            plot_state=PlotState(
                pre_filter={},
                xcol="parent",
                ycol="velocity_mean",
                plot_type=PlotType.SWARM,
                group_col="parent",
            ),
        ),
    )
    monkeypatch.setattr("nicewidgets.nicepool.plot_pool_controller.ui.notify", MagicMock())

    figure_dict = controller._make_figure_dict(controller.plot_states[0])

    assert figure_dict["data"]
    assert controller.plot_states[0].plot_type == PlotType.SWARM
