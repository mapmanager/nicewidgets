"""Tests for ECharts event overlay API."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from nicewidgets.echart_widget.event_overlay import EChartEventOverlayApi
from nicewidgets.echart_widget.models import EChartEventOverlay


class FakeOwner:
    """Fake widget owner recording apply calls."""

    def __init__(self) -> None:
        self.apply_count = 0

    def apply(self) -> None:
        """Record apply call."""
        self.apply_count += 1


@dataclass
class BackendLikeEvent:
    """Event-like object compatible with overlay adapter."""

    id: int
    x0: float
    x1: float
    event_type: str


def test_event_overlay_builds_mark_area_and_selected_style() -> None:
    """Overlay API should build markArea and apply selected style."""
    owner = FakeOwner()
    api = EChartEventOverlayApi(owner)
    api.set_events([BackendLikeEvent(id=2, x0=4.0, x1=1.0, event_type="rise")])
    api.select_event(2)

    mark_area = api.build_mark_area()

    assert owner.apply_count == 2
    assert mark_area is not None
    area = mark_area["data"][0]
    assert area[0]["name"] == "2"
    assert area[0]["xAxis"] == 1.0
    assert area[1]["xAxis"] == 4.0
    assert area[0]["itemStyle"]["borderColor"] == "#ffdd00"


def test_event_overlay_visibility_hides_mark_area() -> None:
    """Hidden overlay API should return an empty markArea to clear ECharts."""
    api = EChartEventOverlayApi(FakeOwner())
    api.add_event(EChartEventOverlay(id="1", x0=0.0, x1=1.0))
    api.set_visible(False)

    assert api.build_mark_area() == {"silent": False, "data": []}


def test_event_overlay_select_unknown_raises() -> None:
    """Selecting an unknown overlay should raise KeyError."""
    api = EChartEventOverlayApi(FakeOwner())
    with pytest.raises(KeyError):
        api.select_event("missing")
