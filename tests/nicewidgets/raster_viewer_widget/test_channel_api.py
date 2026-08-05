"""Tests for the namespaced Python channel API."""

import asyncio
from typing import Any

import pytest

from nicewidgets.raster_viewer_widget.channel_api import ChannelApi
from nicewidgets.raster_viewer_widget.models import RasterChannelDisplay


def test_channel_api_forwards_selection_without_user_event_options() -> None:
    """Verify caller selection reaches the silent component method contract."""
    calls: list[tuple[str, tuple[object, ...]]] = []

    async def run(method: str, *arguments: object) -> Any:
        """Record one simulated component invocation."""
        calls.append((method, arguments))
        return "channel_2"

    result = asyncio.run(ChannelApi(run).select("channel_2"))
    assert result == "channel_2"
    assert calls == [("selectChannel", ("channel_2",))]


def test_channel_api_serializes_complete_typed_display_state() -> None:
    """Verify LUT, explicit contrast, and visibility cross the bridge together."""
    calls: list[tuple[str, tuple[object, ...]]] = []
    applied = {
        "channel_id": "channel_1",
        "lut": "magenta",
        "value_min": 10.0,
        "value_max": 200.0,
        "visible": False,
    }

    async def run(method: str, *arguments: object) -> Any:
        """Record one simulated component invocation."""
        calls.append((method, arguments))
        return applied

    display = RasterChannelDisplay("magenta", 10.0, 200.0, False)
    result = asyncio.run(ChannelApi(run).set_display("channel_1", display))
    assert result == applied
    assert calls == [("setChannelDisplay", ("channel_1", display.to_json()))]


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(1.0, None), (None, 2.0), (2.0, 1.0), (float("nan"), 2.0)],
)
def test_channel_display_rejects_incomplete_or_invalid_ranges(
    minimum: float | None, maximum: float | None
) -> None:
    """Verify invalid contrast state is rejected before reaching JavaScript."""
    with pytest.raises(ValueError):
        RasterChannelDisplay(value_min=minimum, value_max=maximum)
