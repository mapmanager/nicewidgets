"""Tests for PlotlyRasterViewer ``on_viewport_changed``."""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np

if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.ui = types.SimpleNamespace()
    fake_nicegui.app = types.SimpleNamespace(storage=types.SimpleNamespace(general={}))
    fake_nicegui.background_tasks = types.SimpleNamespace(create=lambda *_a, **_k: None)
    sys.modules['nicegui'] = fake_nicegui
else:
    mod = sys.modules['nicegui']
    if not hasattr(mod, 'app'):
        mod.app = types.SimpleNamespace(storage=types.SimpleNamespace(general={}))

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec
from nicewidgets.raster_viewer.frontend.plotly_viewer import DisplayAxisRanges, PlotlyRasterViewer


def _viewer(callback) -> PlotlyRasterViewer:
    viewer = PlotlyRasterViewer(on_viewport_changed=callback)
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='s', y_unit='um')
    asyncio.run(viewer.set_data(data, grid=grid))
    return viewer


def test_viewport_emit_fires_on_new_ranges() -> None:
    seen: list[DisplayAxisRanges] = []
    viewer = _viewer(lambda vp: seen.append(vp))
    viewer._last_emitted_viewport = ((0.0, 10.0), (0.0, 10.0))
    viewer._emit_viewport_from_display(((1.0, 4.0), (2.0, 5.0)))
    assert seen == [((1.0, 4.0), (2.0, 5.0))]


def test_viewport_emit_suppresses_echo() -> None:
    seen: list[DisplayAxisRanges] = []
    viewer = _viewer(lambda vp: seen.append(vp))
    vp: DisplayAxisRanges = ((1.0, 4.0), (2.0, 5.0))
    viewer._last_emitted_viewport = vp
    viewer._emit_viewport_from_display(vp)
    assert seen == []


def test_set_data_pins_last_emitted_viewport() -> None:
    viewer = _viewer(callback=None)
    assert viewer._last_emitted_viewport is not None
    assert viewer._last_emitted_viewport == viewer.get_viewport()
