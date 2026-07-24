"""Tests for pywebview-only Plotly context-menu pointer guards."""

from __future__ import annotations

import types

import pytest

from nicewidgets.raster_viewer.frontend.plotly_context_menu_guards import pywebview_plot_context_menu_guard_js
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer


def test_pywebview_guard_js_is_idempotent_and_blocks_non_primary_buttons() -> None:
    """Guard script should tag the plot div and use capture-phase listeners."""
    js = pywebview_plot_context_menu_guard_js(plot_id=42)

    assert 'getElement(42)' in js
    assert 'csRasterContextMenuGuard' in js
    assert 'ev.button !== 0' in js
    assert 'stopImmediatePropagation' in js
    assert 'preventDefault' in js
    assert 'pointerdown' in js


def test_build_schedules_pywebview_guard_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Desktop builds should schedule guard installation; browser builds should not."""
    scheduled: list[object] = []

    class DummyElement:
        id = 7

        def on(self, *_args, **_kwargs) -> DummyElement:
            return self

    class DummyContextMenu:
        def clear(self) -> DummyContextMenu:
            return self

        def __enter__(self) -> DummyContextMenu:
            return self

        def __exit__(self, *_args) -> None:
            return None

    class DummyUI:
        @staticmethod
        def plotly(_figure):
            return DummyElement()

        @staticmethod
        def context_menu() -> DummyContextMenu:
            return DummyContextMenu()

        @staticmethod
        def timer(_delay: float, callback, *, once: bool = False) -> None:
            scheduled.append(callback)

    import nicewidgets.raster_viewer.frontend.plotly_viewer as plotly_viewer_module

    monkeypatch.setattr(
        plotly_viewer_module,
        'ui',
        types.SimpleNamespace(
            plotly=DummyUI.plotly,
            context_menu=DummyUI.context_menu,
            timer=DummyUI.timer,
        ),
    )
    monkeypatch.setattr(plotly_viewer_module, 'is_pywebview_desktop', lambda: True)

    viewer = PlotlyRasterViewer()
    viewer.build()

    assert len(scheduled) == 1
    assert scheduled[0] == viewer._install_pywebview_context_menu_guards


def test_build_skips_pywebview_guard_install_in_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Browser sessions should not schedule guard installation."""
    scheduled: list[object] = []

    class DummyElement:
        id = 7

        def on(self, *_args, **_kwargs) -> DummyElement:
            return self

    class DummyContextMenu:
        def clear(self) -> DummyContextMenu:
            return self

        def __enter__(self) -> DummyContextMenu:
            return self

        def __exit__(self, *_args) -> None:
            return None

    class DummyUI:
        @staticmethod
        def plotly(_figure):
            return DummyElement()

        @staticmethod
        def context_menu() -> DummyContextMenu:
            return DummyContextMenu()

        @staticmethod
        def timer(_delay: float, callback, *, once: bool = False) -> None:
            scheduled.append(callback)

    import nicewidgets.raster_viewer.frontend.plotly_viewer as plotly_viewer_module

    monkeypatch.setattr(
        plotly_viewer_module,
        'ui',
        types.SimpleNamespace(
            plotly=DummyUI.plotly,
            context_menu=DummyUI.context_menu,
            timer=DummyUI.timer,
        ),
    )
    monkeypatch.setattr(plotly_viewer_module, 'is_pywebview_desktop', lambda: False)

    PlotlyRasterViewer().build()

    assert scheduled == []


def test_install_pywebview_context_menu_guards_runs_guard_js(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard installer should push the guard JavaScript to the browser client."""
    captured: dict[str, object] = {}

    class DummyClient:
        def run_javascript(self, js: str, *, timeout: float = 0) -> None:
            captured['js'] = js
            captured['timeout'] = timeout

    class DummyPlot:
        id = 99
        client = DummyClient()

    monkeypatch.setattr(
        'nicewidgets.raster_viewer.frontend.plotly_viewer.is_pywebview_desktop',
        lambda: True,
    )

    viewer = PlotlyRasterViewer()
    viewer._plot = DummyPlot()
    viewer._install_pywebview_context_menu_guards()

    assert 'getElement(99)' in str(captured.get('js'))
    assert captured.get('timeout') == 2.0
