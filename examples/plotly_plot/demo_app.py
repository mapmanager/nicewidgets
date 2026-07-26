"""NiceGUI PlotlyPlotWidget demo: traces, scatters, range, and measurements.

Run from the repository root:

    uv run python examples/plotly_plot/demo_app.py

Data lives in ``sample_data.py``; widget wiring lives in ``demo_controller.py``;
reusable page composition lives in ``page.py``. This module only registers the
standalone route and starts NiceGUI.
"""

from __future__ import annotations

from nicegui import ui

from nicewidgets.gui_defaults import setUpGuiDefaults
from nicewidgets.utils.logging import setup_logging

try:
    from examples.plotly_plot.page import build_plotly_plot_demo_page
except ImportError:
    # Running as a plain script puts this directory on sys.path.
    from page import build_plotly_plot_demo_page  # type: ignore[no-redef]

setup_logging(level='DEBUG')
setUpGuiDefaults(text_size='text-xs')


@ui.page('/')
def home() -> None:
    """Build the demo page."""
    build_plotly_plot_demo_page()


def main() -> None:
    """Entry point for ``uv run python examples/plotly_plot/demo_app.py``."""
    ui.run(title='nicewidgets PlotlyPlotWidget demo', port=8080, reload=False)


if __name__ in {'__main__', '__mp_main__'}:
    main()
