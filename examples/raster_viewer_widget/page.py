"""Reusable page builder for the canvas RasterViewerWidget demo."""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from examples.raster_viewer_widget.raster_demo import RasterViewerDemo


def build_raster_widget_demo_page(
    *,
    embedded: bool = False,
    dark_mode: bool = True,
) -> RasterViewerDemo:
    """Build the RasterViewerWidget demo in the current NiceGUI slot.

    Args:
        embedded: Whether a parent application owns the page chrome.
        dark_mode: Initial viewer theme.

    Returns:
        Themeable demo handle containing the mounted viewer.
    """
    example_root = Path(__file__).resolve().parent
    demo = RasterViewerDemo(example_root)
    height_class = "h-full min-h-0" if embedded else "min-h-screen"
    with ui.column().classes(f"w-full {height_class} gap-2 p-4"):
        demo.build_page(embedded=embedded, dark_mode=dark_mode)
    return demo
