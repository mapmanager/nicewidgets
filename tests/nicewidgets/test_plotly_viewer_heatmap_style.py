"""Tests for :meth:`PlotlyRasterViewer.set_heatmap_style` combined call."""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np

if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.ui = types.SimpleNamespace()
    sys.modules['nicegui'] = fake_nicegui

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer


class _FakeClient:
    def __init__(self) -> None:
        self.js_calls: list[str] = []

    def run_javascript(self, js: str, *, timeout: float = 2.0) -> None:
        _ = timeout
        self.js_calls.append(js)


class _FakePlot:
    """Just enough of a NiceGUI Plotly element for ``set_heatmap_style`` paths."""

    def __init__(self) -> None:
        self.id = 'fake-plot'
        self.client = _FakeClient()


def _viewer_with_heatmap_trace() -> PlotlyRasterViewer:
    """Build a viewer with a heatmap trace and a fake browser plot bound.

    The viewer never gets a real NiceGUI plot; we install a tiny stub so the
    ``_heatmap_trace_active()`` branch runs and the JS restyle is captured
    on ``_plot.client.js_calls``.
    """
    viewer = PlotlyRasterViewer()
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='s', y_unit='um')
    asyncio.run(viewer.set_data(data, grid=grid))
    viewer._plot = _FakePlot()  # type: ignore[assignment]
    viewer._plotly_dict['data'] = [{'type': 'heatmap'}]
    return viewer


def test_set_heatmap_style_writes_all_three_keys_into_local_figure() -> None:
    """A single combined call updates ``colorscale`` + ``zmin`` + ``zmax`` atomically."""
    viewer = _viewer_with_heatmap_trace()
    client = viewer._plot.client  # type: ignore[union-attr]

    asyncio.run(
        viewer.set_heatmap_style(
            colorscale='Plasma',
            zmin=12.0,
            zmax=212.0,
        )
    )

    trace0 = viewer._plotly_dict['data'][0]
    assert trace0['colorscale'] == 'Plasma'
    assert trace0['zmin'] == 12.0
    assert trace0['zmax'] == 212.0
    assert viewer._heatmap_colorscale == 'Plasma'
    assert viewer._contrast_zmin == 12.0
    assert viewer._contrast_zmax == 212.0
    # Exactly one ``Plotly.restyle`` issued.
    assert len(client.js_calls) == 1
    js = client.js_calls[0]
    assert 'Plotly.restyle' in js
    assert '"Plasma"' in js
    assert '12.0' in js
    assert '212.0' in js


def test_set_heatmap_style_swaps_inverted_zmin_zmax() -> None:
    """Inverted ``zmin``/``zmax`` are normalized to ``(min, max)``."""
    viewer = _viewer_with_heatmap_trace()

    asyncio.run(
        viewer.set_heatmap_style(colorscale='Viridis', zmin=200.0, zmax=10.0)
    )

    trace0 = viewer._plotly_dict['data'][0]
    assert trace0['zmin'] == 10.0
    assert trace0['zmax'] == 200.0


def test_set_heatmap_style_accepts_list_form_colorscale() -> None:
    """The 2-stop list form used by ``inverted_grays`` propagates through."""
    viewer = _viewer_with_heatmap_trace()
    client = viewer._plot.client  # type: ignore[union-attr]

    scale = [[0, 'rgb(255,255,255)'], [1, 'rgb(0,0,0)']]
    asyncio.run(
        viewer.set_heatmap_style(colorscale=scale, zmin=0.0, zmax=255.0)
    )

    trace0 = viewer._plotly_dict['data'][0]
    assert trace0['colorscale'] == scale
    assert viewer._heatmap_colorscale == scale
    js = client.js_calls[-1]
    assert 'rgb(255,255,255)' in js
    assert 'rgb(0,0,0)' in js


def test_set_heatmap_style_skips_restyle_when_values_already_match() -> None:
    """Idempotent: a no-op repeat does not issue a second ``Plotly.restyle``."""
    viewer = _viewer_with_heatmap_trace()
    client = viewer._plot.client  # type: ignore[union-attr]

    asyncio.run(viewer.set_heatmap_style(colorscale='Plasma', zmin=12.0, zmax=212.0))
    asyncio.run(viewer.set_heatmap_style(colorscale='Plasma', zmin=12.0, zmax=212.0))

    assert len(client.js_calls) == 1
