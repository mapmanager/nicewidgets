"""Standalone ``TableWidget`` demo entry point.

Run from the repository root:

    uv run python examples/table_widget/demo_app.py

Reusable page composition lives in ``page.py``. This module only registers the
standalone route and starts NiceGUI.
"""

from __future__ import annotations

from nicegui import ui

from nicewidgets.gui_defaults import setUpGuiDefaults

try:
    from examples.table_widget.page import build_table_demo_page
except ImportError:
    # Running as a plain script puts this directory on sys.path.
    from page import build_table_demo_page  # type: ignore[no-redef]

setUpGuiDefaults(text_size='text-xs')

@ui.page('/')
def home_page() -> None:
    """NiceGUI home page: single demo table."""
    build_table_demo_page()


def main() -> None:
    """Entry point for ``uv run python examples/table_widget/demo_app.py``."""
    ui.run(title='nicewidgets TableWidget demo', port=8080, reload=False)


if __name__ == '__main__':
    main()
