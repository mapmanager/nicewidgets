"""Clipboard helpers for ECharts widget exports.

These helpers use ``ui.echart.run_chart_method('getDataURL', ...)`` — the
NiceGUI-documented way to invoke a method on the underlying ECharts
instance — rather than reaching into the DOM with
``window.echarts.getInstanceByDom``. That keeps the helpers robust against
NiceGUI version changes and avoids the ``chart_not_ready`` failure that
``getInstanceByDom`` produces when the Vue host element is not the same as
the chart-mount div.
"""

from __future__ import annotations

import base64
import binascii
import json
from typing import TYPE_CHECKING

from nicegui import ui

from nicewidgets.utils.logging import get_logger

if TYPE_CHECKING:
    from nicegui.element import Element

logger = get_logger(__name__)


_DATA_URL_PNG_PREFIX = "data:image/png;base64,"
_GET_DATA_URL_OPTIONS = {
    "type": "png",
    "pixelRatio": 2,
    "backgroundColor": "#ffffff",
}


async def _get_data_url(echart_element: Element) -> str:
    """Return the chart's PNG data URL via NiceGUI's chart-method bridge.

    Args:
        echart_element: NiceGUI ``ui.echart`` element to export.

    Returns:
        Base64-encoded PNG data URL.

    Raises:
        RuntimeError: If the chart instance method is unavailable.
        ValueError: If the chart does not return a PNG data URL.
    """
    try:
        data_url = await echart_element.run_chart_method(
            "getDataURL", _GET_DATA_URL_OPTIONS, timeout=10.0
        )
    except TimeoutError as exc:
        raise RuntimeError("ECharts getDataURL timed out.") from exc
    if not isinstance(data_url, str) or not data_url.startswith(_DATA_URL_PNG_PREFIX):
        raise ValueError(
            f"ECharts getDataURL did not return a PNG data URL (got: {data_url!r})."
        )
    return data_url


async def get_echart_png_bytes(echart_element: Element) -> bytes:
    """Return PNG bytes for a NiceGUI ECharts element.

    Args:
        echart_element: NiceGUI ``ui.echart`` element to export.

    Returns:
        PNG image bytes.

    Raises:
        RuntimeError: If the browser-side export fails.
        ValueError: If the chart does not return a PNG data URL.
    """
    data_url = await _get_data_url(echart_element)
    b64 = data_url.split(",", 1)[1]
    try:
        png_bytes = base64.b64decode(b64, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"Invalid base64 PNG data: {exc}") from exc
    logger.info("Exported ECharts PNG: %d bytes", len(png_bytes))
    return png_bytes


async def copy_echart_png_to_browser_clipboard(echart_element: Element) -> None:
    """Copy an ECharts PNG export to the browser clipboard.

    Args:
        echart_element: NiceGUI ``ui.echart`` element to export.

    Raises:
        RuntimeError: If the Clipboard API is unavailable or copying fails.
    """
    data_url = await _get_data_url(echart_element)
    js = f"""
(async () => {{
  const out = {{ ok: false, stage: 'start' }};
  try {{
    if (!navigator.clipboard || typeof ClipboardItem === 'undefined') {{
      out.stage = 'clipboard_unavailable';
      return out;
    }}
    const dataUrl = {json.dumps(data_url)};
    const blob = await (await fetch(dataUrl)).blob();
    await navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]);
    out.ok = true;
    out.stage = 'done';
    return out;
  }} catch (err) {{
    out.stage = 'error';
    out.error = String(err);
    return out;
  }}
}})()
"""
    result = await ui.run_javascript(js, timeout=10.0)
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(f"Browser clipboard copy failed: {result}")
