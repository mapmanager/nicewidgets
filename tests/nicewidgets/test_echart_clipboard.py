"""Tests for :mod:`nicewidgets.echart_widget.clipboard`.

These tests verify the source-of-truth contract for PNG export: the helpers
must use ``ui.echart.run_chart_method('getDataURL', ...)`` (NiceGUI's
documented bridge to the underlying ECharts instance) rather than reaching
into the DOM via ``window.echarts.getInstanceByDom``. The previous DOM-lookup
path raised ``chart_not_ready`` because NiceGUI's element ``$el`` is the Vue
root, not the chart-mount div.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from nicewidgets.echart_widget import clipboard as clipboard_mod


class _FakeEchartElement:
    """Capture ``run_chart_method`` calls and return a scripted PNG data URL."""

    def __init__(self, data_url: str | None = None, *, raise_error: bool = False) -> None:
        self.data_url = data_url
        self.raise_error = raise_error
        self.calls: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    async def run_chart_method(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((name, args[0] if args else {}, kwargs))
        if self.raise_error:
            raise TimeoutError("simulated timeout")
        return self.data_url


def _valid_data_url() -> str:
    # 1x1 transparent PNG; payload bytes are irrelevant for the unit test.
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNgAAIAAAUAAen63NgAAAAASUVORK5CYII="


def test_get_data_url_invokes_run_chart_method_with_documented_options() -> None:
    """``_get_data_url`` must call ``run_chart_method('getDataURL', ...)``."""
    fake = _FakeEchartElement(data_url=_valid_data_url())
    result = asyncio.run(clipboard_mod._get_data_url(fake))

    assert result == _valid_data_url()
    assert len(fake.calls) == 1
    name, opts, _ = fake.calls[0]
    assert name == "getDataURL"
    assert opts == clipboard_mod._GET_DATA_URL_OPTIONS
    # The options dict matches ECharts' documented getDataURL signature.
    assert opts["type"] == "png"
    assert opts["pixelRatio"] == 2
    assert "backgroundColor" in opts


def test_get_data_url_rejects_non_png_response() -> None:
    """A non-PNG data URL is treated as a contract violation."""
    fake = _FakeEchartElement(data_url="data:image/jpeg;base64,Zm9v")
    with pytest.raises(ValueError, match="PNG data URL"):
        asyncio.run(clipboard_mod._get_data_url(fake))


def test_get_data_url_rejects_non_string_response() -> None:
    """A non-string result is rejected (defensive, not expected at runtime)."""
    fake = _FakeEchartElement(data_url=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="PNG data URL"):
        asyncio.run(clipboard_mod._get_data_url(fake))


def test_get_data_url_wraps_timeout_error_as_runtime_error() -> None:
    """A timeout from ``run_chart_method`` is reported as a clear RuntimeError."""
    fake = _FakeEchartElement(raise_error=True)
    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(clipboard_mod._get_data_url(fake))


def test_get_echart_png_bytes_decodes_base64_payload() -> None:
    """``get_echart_png_bytes`` returns the decoded PNG bytes."""
    fake = _FakeEchartElement(data_url=_valid_data_url())
    png_bytes = asyncio.run(clipboard_mod.get_echart_png_bytes(fake))
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_browser_copy_uses_run_chart_method_then_runs_clipboard_js(monkeypatch) -> None:
    """``copy_echart_png_to_browser_clipboard`` calls run_chart_method then JS."""
    fake = _FakeEchartElement(data_url=_valid_data_url())

    js_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_run_javascript(js: str, *, timeout: float = 10.0) -> dict[str, Any]:
        js_calls.append((js, {"timeout": timeout}))
        return {"ok": True}

    monkeypatch.setattr(clipboard_mod.ui, "run_javascript", fake_run_javascript)

    asyncio.run(clipboard_mod.copy_echart_png_to_browser_clipboard(fake))

    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "getDataURL"
    assert len(js_calls) == 1
    js_source = js_calls[0][0]
    # The JS snippet must use navigator.clipboard.write with a ClipboardItem.
    assert "navigator.clipboard" in js_source
    assert "ClipboardItem" in js_source


def test_browser_copy_raises_when_js_reports_failure(monkeypatch) -> None:
    """A non-ok JS result is reported as ``RuntimeError`` so the widget can notify."""
    fake = _FakeEchartElement(data_url=_valid_data_url())

    async def fake_run_javascript(js: str, *, timeout: float = 10.0) -> dict[str, Any]:
        return {"ok": False, "stage": "clipboard_unavailable"}

    monkeypatch.setattr(clipboard_mod.ui, "run_javascript", fake_run_javascript)

    with pytest.raises(RuntimeError, match="Browser clipboard copy failed"):
        asyncio.run(clipboard_mod.copy_echart_png_to_browser_clipboard(fake))


def test_clipboard_module_does_not_query_dom_for_chart_instance() -> None:
    """The new helpers must not rely on ``window.echarts.getInstanceByDom``.

    This is the regression guard for the original ``chart_not_ready`` error
    that surfaced when the previous implementation queried the DOM directly.
    The check inspects each helper's body so module-level docstring references
    do not cause false positives.
    """
    import inspect

    bodies = [
        inspect.getsource(clipboard_mod._get_data_url),
        inspect.getsource(clipboard_mod.get_echart_png_bytes),
        inspect.getsource(clipboard_mod.copy_echart_png_to_browser_clipboard),
    ]
    for src in bodies:
        assert "getInstanceByDom" not in src
        assert "window.echarts" not in src
