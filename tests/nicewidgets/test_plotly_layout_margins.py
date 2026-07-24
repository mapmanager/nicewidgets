"""Tests for shared Plotly layout margin profiles."""

from __future__ import annotations

from nicewidgets.plotly_layout_margins import PlotlyLayoutMarginsProfile


def test_profile_resolve_switches_between_compact_and_labeled() -> None:
    """Profile should return the configured margin dict for each axis-label state."""
    profile = PlotlyLayoutMarginsProfile(
        with_axis_labels={"l": 60, "r": 24, "t": 10, "b": 40},
        compact={"l": 8, "r": 8, "t": 8, "b": 8},
    )

    assert profile.resolve(show_axis_labels=True) == {"l": 60, "r": 24, "t": 10, "b": 40}
    assert profile.resolve(show_axis_labels=False) == {"l": 8, "r": 8, "t": 8, "b": 8}


def test_profile_apply_axis_stabilization_disables_automargin() -> None:
    """Stack profiles should pin primary axis automargin off."""
    profile = PlotlyLayoutMarginsProfile(
        with_axis_labels={"l": 60, "r": 24, "t": 10, "b": 40},
        compact={"l": 8, "r": 8, "t": 8, "b": 8},
        stabilize_axis_automargin=True,
    )
    layout: dict[str, object] = {"xaxis": {}, "yaxis": {}}

    profile.apply_axis_stabilization(layout)

    assert layout["xaxis"] == {"automargin": False}
    assert layout["yaxis"] == {"automargin": False}
