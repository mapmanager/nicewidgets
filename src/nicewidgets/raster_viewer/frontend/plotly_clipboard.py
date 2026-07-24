"""Clipboard helpers for Plotly raster viewer exports."""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING

from nicegui import ui

from nicewidgets.utils.clipboard import copy_png_bytes_to_native_clipboard
from nicewidgets.utils.logging import get_logger

if TYPE_CHECKING:
    from nicegui.element import Element

logger = get_logger(__name__)


__all__ = [
    "copy_plotly_png_to_browser_clipboard",
    "copy_png_bytes_to_native_clipboard",
    "get_plotly_png_bytes",
    "json_or_null",
]


async def get_plotly_png_bytes(
    plot_widget: Element,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: int = 1,
) -> bytes:
    """Return PNG bytes for a NiceGUI Plotly element using browser Plotly export.

    Args:
        plot_widget: NiceGUI Plotly element to export.
        width: Optional output width in pixels. Defaults to rendered width.
        height: Optional output height in pixels. Defaults to rendered height.
        scale: Plotly export scale factor.

    Returns:
        PNG image bytes.

    Raises:
        RuntimeError: If browser-side export fails.
        ValueError: If the browser does not return a PNG data URL.
    """
    js = f"""
(async () => {{
  const out = {{ ok: false, stage: 'start' }};
  try {{
    const host = getElement({plot_widget.id}).$el;
    if (!host) {{
      out.stage = 'no_host';
      return out;
    }}
    const plotDiv = host.querySelector('.js-plotly-plot') || host;
    if (!plotDiv || !plotDiv._fullLayout) {{
      out.stage = 'plot_not_ready';
      return out;
    }}
    const exportWidth = {json_or_null(width)} ?? Math.round(plotDiv.offsetWidth);
    const exportHeight = {json_or_null(height)} ?? Math.round(plotDiv.offsetHeight);
    out.stage = 'to_image';
    const dataUrl = await Plotly.toImage(plotDiv, {{
      format: 'png',
      width: exportWidth,
      height: exportHeight,
      scale: {int(scale)},
    }});
    out.ok = true;
    out.stage = 'done';
    out.width = exportWidth;
    out.height = exportHeight;
    out.data_url = dataUrl;
    return out;
  }} catch (err) {{
    out.stage = 'error';
    out.error = String(err);
    return out;
  }}
}})()
"""
    result = await ui.run_javascript(js, timeout=30.0)
    if not isinstance(result, dict) or not result.get('ok'):
        raise RuntimeError(f'Plot export failed: {result}')

    data_url = result.get('data_url')
    if not isinstance(data_url, str) or not data_url.startswith('data:image/png;base64,'):
        raise ValueError('JavaScript did not return a PNG data URL.')

    b64 = data_url.split(',', 1)[1]
    try:
        png_bytes = base64.b64decode(b64, validate=True)
    except binascii.Error as exc:
        raise ValueError(f'Invalid base64 PNG data: {exc}') from exc

    logger.info('Exported Plotly PNG: %d bytes', len(png_bytes))
    return png_bytes



async def copy_plotly_png_to_browser_clipboard(
    plot_widget: Element,
    *,
    width: int | None = None,
    height: int | None = None,
    scale: int = 1,
) -> None:
    """Copy a Plotly PNG export to the browser clipboard.

    Args:
        plot_widget: NiceGUI Plotly element to export.
        width: Optional output width in pixels. Defaults to rendered width.
        height: Optional output height in pixels. Defaults to rendered height.
        scale: Plotly export scale factor.

    Raises:
        RuntimeError: If the Clipboard API is unavailable or copying fails.
    """
    js = f"""
(async () => {{
  const out = {{ ok: false, stage: 'start' }};
  try {{
    if (!navigator.clipboard || typeof ClipboardItem === 'undefined') {{
      out.stage = 'clipboard_unavailable';
      return out;
    }}
    const host = getElement({plot_widget.id}).$el;
    if (!host) {{
      out.stage = 'no_host';
      return out;
    }}
    const plotDiv = host.querySelector('.js-plotly-plot') || host;
    if (!plotDiv || !plotDiv._fullLayout) {{
      out.stage = 'plot_not_ready';
      return out;
    }}
    const exportWidth = {json_or_null(width)} ?? Math.round(plotDiv.offsetWidth);
    const exportHeight = {json_or_null(height)} ?? Math.round(plotDiv.offsetHeight);
    const dataUrl = await Plotly.toImage(plotDiv, {{
      format: 'png',
      width: exportWidth,
      height: exportHeight,
      scale: {int(scale)},
    }});
    const blob = await (await fetch(dataUrl)).blob();
    await navigator.clipboard.write([new ClipboardItem({{'image/png': blob}})]);
    out.ok = true;
    out.stage = 'done';
    out.width = exportWidth;
    out.height = exportHeight;
    return out;
  }} catch (err) {{
    out.stage = 'error';
    out.error = String(err);
    return out;
  }}
}})()
"""
    result = await ui.run_javascript(js, timeout=30.0)
    if not isinstance(result, dict) or not result.get('ok'):
        raise RuntimeError(f'Browser clipboard copy failed: {result}')


def json_or_null(value: int | None) -> str:
    """Return an integer JavaScript literal or ``null``.

    Args:
        value: Optional integer value.

    Returns:
        JavaScript literal string.
    """
    return 'null' if value is None else str(int(value))
