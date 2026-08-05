"""Tests for the typed and namespaced Python X/Y plot API."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from nicewidgets.raster_viewer_widget import XYPlot, XYPlotMode, XYPlotStyle
from nicewidgets.raster_viewer_widget.xy_plot_api import XYPlotApi


def test_xy_plot_serializes_nonfinite_coordinates_as_json_gaps() -> None:
    """Preserve point indices while making NumPy non-finite values JSON-safe."""
    plot = XYPlot(
        plot_id="cells",
        name="Cell locations",
        x=np.array([-1.0, np.nan, 8.0]),
        y=np.array([2.0, 4.0, np.inf]),
        point_ids=("a", "b", "c"),
        mode=XYPlotMode.LINES_MARKERS,
        style=XYPlotStyle(color="#00ffff", marker_size=7),
    )
    payload = plot.to_json()
    assert payload["x"] == [-1.0, None, 8.0]
    assert payload["y"] == [2.0, 4.0, None]
    assert payload["point_ids"] == ["a", "b", "c"]
    assert payload["coordinate_space"] == "physical"


def test_xy_plot_rejects_inconsistent_identity_and_coordinates() -> None:
    """Fail before crossing the browser bridge when plot data is inconsistent."""
    with pytest.raises(ValueError, match="equal lengths"):
        XYPlot(plot_id="bad", x=(1.0,), y=(1.0, 2.0))
    with pytest.raises(ValueError, match="same length"):
        XYPlot(plot_id="bad", x=(1.0,), y=(2.0,), point_ids=("a", "b"))
    with pytest.raises(ValueError, match="physical"):
        XYPlot(plot_id="bad", x=(), y=(), coordinate_space="pixels")


def test_namespaced_xy_plot_api_calls_flat_component_methods() -> None:
    """Keep Python organization independent of NiceGUI's flat method bridge."""
    calls: list[tuple[str, tuple[object, ...]]] = []

    async def run_method(method: str, *arguments: object) -> Any:
        calls.append((method, arguments))
        return "sample" if method == "addXYPlot" else True

    async def exercise() -> None:
        api = XYPlotApi(run_method)
        plot = XYPlot(plot_id="sample", x=(1.0,), y=(2.0,))
        assert await api.add(plot) == "sample"
        assert await api.update(plot)
        assert await api.hide("sample")
        assert await api.show("sample")
        assert await api.remove("sample")

    asyncio.run(exercise())
    assert [method for method, _arguments in calls] == [
        "addXYPlot",
        "updateXYPlot",
        "hideXYPlot",
        "showXYPlot",
        "removeXYPlot",
    ]
