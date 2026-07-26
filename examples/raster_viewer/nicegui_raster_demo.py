"""NiceGUI raster viewer demo: PlotlyRasterViewer + ImageToolbarWidget + ContrastWidget.

Run from the repository root:

    uv run python examples/raster_viewer/nicegui_raster_demo.py

Layout and wiring live in ``demo_controller.py``; synthetic datasets live in
``sample_data.py``; reusable page composition lives in ``page.py``. This entry
module only registers the standalone route and starts NiceGUI.
"""

from __future__ import annotations

from nicegui import ui

from nicewidgets.gui_defaults import setUpGuiDefaults
from nicewidgets.utils.logging import setup_logging

try:
    from examples.raster_viewer.page import build_raster_demo_page
except ImportError:
    # Running as a plain script puts this directory on sys.path.
    from page import build_raster_demo_page  # type: ignore[no-redef]

setup_logging(level='DEBUG')
setUpGuiDefaults(text_size='text-xs')


@ui.page('/')
def home() -> None:
    """Build the demo page."""
    build_raster_demo_page()


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(reload=False, native=True)
