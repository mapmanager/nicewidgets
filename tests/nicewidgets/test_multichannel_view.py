"""Tests for MultiChannelRasterView coordinator (Phase 2)."""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

class _FakeCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def classes(self, *_a, **_k):
        return self

    def clear(self):
        return None

    def on(self, *_a, **_k):
        return None


class _FakePlot(_FakeCtx):
    def __init__(self):
        self.id = 1
        self.figure = {}
        self.client = types.SimpleNamespace(run_javascript=lambda *a, **k: None)

    def update(self):
        return None


def _ensure_nicegui_stub() -> None:
    """Install or repair a NiceGUI stub (other tests may leave a partial fake)."""
    fake_ui = types.SimpleNamespace(
        column=lambda *a, **k: _FakeCtx(),
        row=lambda *a, **k: _FakeCtx(),
        element=lambda *a, **k: _FakeCtx(),
        label=lambda *a, **k: _FakeCtx(),
        context_menu=lambda *a, **k: _FakeCtx(),
        plotly=lambda *a, **k: _FakePlot(),
        timer=lambda *a, **k: None,
    )
    bg = types.SimpleNamespace(
        create=lambda coro: asyncio.run(coro) if asyncio.iscoroutine(coro) else None
    )
    mod = sys.modules.get('nicegui')
    if mod is None or getattr(mod, '__file__', None) is None:
        # Replace incomplete fakes; keep a real installed nicegui if present.
        if mod is None or not hasattr(mod, 'app'):
            mod = types.ModuleType('nicegui')
            sys.modules['nicegui'] = mod
    if not hasattr(mod, 'ui'):
        mod.ui = fake_ui
    if not hasattr(mod, 'background_tasks'):
        mod.background_tasks = bg
    if not hasattr(mod, 'app'):
        mod.app = types.SimpleNamespace(storage=types.SimpleNamespace(general={}))


_ensure_nicegui_stub()

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec
from nicewidgets.raster_viewer.frontend.roi_overlay import RectRoiOverlay
from nicewidgets.raster_viewer.multichannel.models import (
    ChannelDisplayStyle,
    ChannelPlane,
    MultiChannelRasterViewConfig,
)
from nicewidgets.raster_viewer.multichannel.view import MultiChannelRasterView


def _planes(n: int = 2, shape: tuple[int, int] = (8, 8)) -> list[ChannelPlane]:
    out: list[ChannelPlane] = []
    for i in range(n):
        data = np.full(shape, float(i + 1) * 10.0, dtype=np.float32)
        out.append(ChannelPlane(channel_id=i, data=data, label=str(i)))
    return out


def test_set_layout_mode_composite_not_implemented() -> None:
    view = MultiChannelRasterView()
    with pytest.raises(NotImplementedError, match='Phase 3'):
        asyncio.run(view.set_layout_mode('composite'))


def test_set_link_viewport_updates_config() -> None:
    view = MultiChannelRasterView(config=MultiChannelRasterViewConfig(link_viewport=True))
    assert view.config.link_viewport is True
    view.set_link_viewport(False)
    assert view.config.link_viewport is False


def test_wait_for_panes_ready_timeout_without_ui_timer() -> None:
    """Ready-wait must not create NiceGUI timers (background-task safe)."""
    view = MultiChannelRasterView()
    view._pane_plots = {0: object()}  # type: ignore[assignment]
    # No afterplot resolution → times out cleanly.
    asyncio.run(view._wait_for_panes_ready())


def test_visible_planes_single_vs_mosaic() -> None:
    view = MultiChannelRasterView()
    view._planes = _planes(2)
    view._active_channel_id = 1
    view._config = MultiChannelRasterViewConfig(layout_mode='single')
    assert [p.channel_id for p in view._visible_planes_for_layout()] == [1]

    view._config = MultiChannelRasterViewConfig(layout_mode='mosaic')
    assert [p.channel_id for p in view._visible_planes_for_layout()] == [0, 1]

    hidden = [
        ChannelPlane(
            channel_id=0,
            data=np.zeros((4, 4), dtype=np.float32),
            style=ChannelDisplayStyle(visible=False),
        ),
        ChannelPlane(channel_id=1, data=np.ones((4, 4), dtype=np.float32)),
    ]
    view._planes = hidden
    assert [p.channel_id for p in view._visible_planes_for_layout()] == [1]


def test_set_rois_fans_out_to_mock_viewers() -> None:
    view = MultiChannelRasterView()

    @dataclass
    class _V:
        rois: list[Any] = field(default_factory=list)
        selected: int | None = None

        def set_rois(self, rois):
            self.rois = list(rois)

        def select_roi(self, roi_id):
            self.selected = roi_id

        def add_roi(self, roi):
            self.rois = [r for r in self.rois if r.roi_id != roi.roi_id] + [roi]

        def delete_roi(self, roi_id):
            self.rois = [r for r in self.rois if r.roi_id != roi_id]

    a, b = _V(), _V()
    view._viewers = {0: a, 1: b}  # type: ignore[assignment]
    roi = RectRoiOverlay(roi_id=1, x0=0, x1=1, y0=0, y1=1)
    view.set_rois([roi])
    assert a.rois == [roi] and b.rois == [roi]
    view.select_roi(1)
    assert a.selected == 1 and b.selected == 1


def test_set_channels_requires_same_shape() -> None:
    view = MultiChannelRasterView()
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='', y_unit='')
    planes = [
        ChannelPlane(channel_id=0, data=np.zeros((4, 4), dtype=np.float32)),
        ChannelPlane(channel_id=1, data=np.zeros((4, 5), dtype=np.float32)),
    ]
    with pytest.raises(ValueError, match='same shape'):
        asyncio.run(view.set_channels(planes, grid=grid))


def test_set_active_channel_rejects_unknown() -> None:
    view = MultiChannelRasterView()
    view._planes = _planes(2)
    with pytest.raises(ValueError, match='unknown channel'):
        asyncio.run(view.set_active_channel(9))
