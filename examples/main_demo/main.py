"""Combined, browser-based NiceWidgets demo application.

Run from the repository root:

    uv run python -m examples.main_demo.main

The three product demos remain independently runnable. Their reusable page
builders are mounted here on separate routes so each heavy visualization owns
an isolated browser page and can be deep-linked.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from nicegui import ui

from examples.nicepool.page import build_nicepool_demo_page
from examples.raster_viewer.page import build_raster_demo_page
from examples.table_widget.page import build_table_demo_page
from nicewidgets.gui_defaults import setUpGuiDefaults
from nicewidgets.utils.logging import setup_logging

setup_logging(level='DEBUG')
setUpGuiDefaults(text_size='text-xs')

DEMO_ROUTES: tuple[tuple[str, str, str], ...] = (
    (
        '/raster',
        'Raster Viewer',
        'Multichannel raster display with toolbar, contrast, and ROI controls.',
    ),
    (
        '/table',
        'TableWidget',
        'Grouped AG Grid table with selection, editing, and context actions.',
    ),
    (
        '/nicepool',
        'NicePool',
        'DataFrame-driven interactive plots with filters and linked selection.',
    ),
)


def build_navigation() -> None:
    """Build the shared route navigation bar."""
    with ui.row().classes(
        'w-full shrink-0 items-center gap-2 px-4 py-2 bg-grey-2'
    ):
        ui.link('NiceWidgets', '/').classes('text-subtitle1 no-underline')
        for path, title, _description in DEMO_ROUTES:
            ui.link(title, path).classes('no-underline')


def build_demo_route(builder: Callable[..., object]) -> None:
    """Build shared navigation followed by an embedded demo page."""
    with ui.column().classes('w-full h-screen min-h-0 gap-0'):
        build_navigation()
        with ui.column().classes('w-full flex-1 min-h-0 overflow-auto gap-0'):
            builder(embedded=True)


@ui.page('/')
def home_page() -> None:
    """Build the combined demo index."""
    with ui.column().classes('w-full min-h-screen gap-6 p-6'):
        ui.label('NiceWidgets Demos').classes('text-h4')
        ui.label(
            'Reusable NiceGUI widgets for scientific and desktop applications.'
        ).classes('text-subtitle1 text-grey-7')

        with ui.row().classes('w-full gap-4 items-stretch flex-wrap'):
            for path, title, description in DEMO_ROUTES:
                with ui.card().classes('w-80 p-4 gap-3'):
                    ui.label(title).classes('text-h6')
                    ui.label(description).classes('text-grey-7 grow')
                    ui.link('Open demo', path).classes('no-underline')


@ui.page('/raster')
def raster_page() -> None:
    """Build the raster viewer route."""
    build_demo_route(build_raster_demo_page)


@ui.page('/table')
def table_page() -> None:
    """Build the table widget route."""
    build_demo_route(build_table_demo_page)


@ui.page('/nicepool')
def nicepool_page() -> None:
    """Build the NicePool route."""
    build_demo_route(build_nicepool_demo_page)


def main() -> None:
    """Run the combined demo in a browser-friendly server configuration."""
    ui.run(
        title='NiceWidgets demos',
        host=os.getenv('NICEWIDGETS_DEMO_HOST', '127.0.0.1'),
        port=int(os.getenv('PORT', '8080')),
        reload=False,
    )


if __name__ in {'__main__', '__mp_main__'}:
    main()
