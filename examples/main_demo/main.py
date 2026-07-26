"""Combined, browser-based NiceWidgets demo application.

Run from the repository root:

    uv run python -m examples.main_demo.main

Product demos remain independently runnable. Their reusable page builders are
mounted here on separate routes so each heavy visualization owns an isolated
browser page and can be deep-linked. The shared top toolbar owns app chrome
(``ui.dark_mode``) and pushes theme into each page's widget API.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Protocol

from nicegui import ui

from examples.nicepool.page import build_nicepool_demo_page
from examples.plotly_plot.page import build_plotly_plot_demo_page
from examples.raster_viewer.page import build_raster_demo_page
from examples.table_widget.page import build_table_demo_page
from examples.tree_widget.page import build_tree_demo_page
from nicewidgets.gui_defaults import setUpGuiDefaults
from nicewidgets.utils.logging import setup_logging

setup_logging(level='DEBUG')
setUpGuiDefaults(text_size='text-xs')

DOCS_URL = 'https://mapmanager.github.io/nicewidgets/'

# Process-local theme for this showcase app. Deep links rebuild the active page
# with the current value; live toggles update the active page handle only.
_APP_DARK_MODE = True

DEMO_ROUTES: tuple[tuple[str, str, str], ...] = (
    (
        '/raster',
        'Raster Viewer',
        'Multichannel raster display with toolbar, contrast, and ROI controls.',
    ),
    (
        '/plotly',
        'PlotlyPlotWidget',
        'Scientific traces, scatters, measurements, and x-range callbacks.',
    ),
    (
        '/nicepool',
        'NicePool',
        'DataFrame-driven interactive plots with filters and linked selection.',
    ),
    (
        '/table',
        'TableWidget',
        'Grouped AG Grid table with selection, editing, and context actions.',
    ),
    (
        '/tree',
        'TreeWidget',
        'AG Grid Enterprise tree with selection and expand/collapse.',
    ),
)


class ThemeablePage(Protocol):
    """Minimal theme handle returned by demo page builders."""

    def set_dark_mode(self, enabled: bool) -> None:
        """Apply light/dark theme to the page's primary widget(s)."""


def _set_app_dark_mode(enabled: bool, *, page: ThemeablePage | None = None) -> None:
    """Update process theme, Quasar chrome, and the active page widget."""
    global _APP_DARK_MODE
    _APP_DARK_MODE = bool(enabled)
    dark = ui.dark_mode()
    if _APP_DARK_MODE:
        dark.enable()
    else:
        dark.disable()
    if page is not None:
        page.set_dark_mode(_APP_DARK_MODE)


def build_toolbar(*, page: ThemeablePage | None = None) -> None:
    """Build the shared top toolbar for home and demo routes.

    Args:
        page: Optional active demo page handle that receives live theme changes.
    """
    with ui.row().classes(
        'w-full shrink-0 items-center gap-3 px-4 py-2 bg-grey-2'
    ):
        ui.link('NiceWidgets', '/').classes('text-subtitle1 no-underline')
        for path, title, _description in DEMO_ROUTES:
            ui.link(title, path).classes('no-underline')
        ui.space()
        ui.switch(
            'Dark',
            value=_APP_DARK_MODE,
            on_change=lambda e: _set_app_dark_mode(bool(e.value), page=page),
        )
        ui.link('Docs', DOCS_URL, new_tab=True).classes('no-underline')


def build_demo_route(builder: Callable[..., Any]) -> None:
    """Build shared toolbar followed by an embedded demo page."""
    # Create dark_mode element early so Quasar chrome matches process state.
    ui.dark_mode(value=_APP_DARK_MODE)
    with ui.column().classes('w-full h-screen min-h-0 gap-0'):
        # Build page first so the toolbar can bind live theme updates to it.
        page_holder: dict[str, ThemeablePage | None] = {'page': None}

        def on_theme_change(enabled: bool) -> None:
            _set_app_dark_mode(enabled, page=page_holder['page'])

        with ui.row().classes(
            'w-full shrink-0 items-center gap-3 px-4 py-2 bg-grey-2'
        ):
            ui.link('NiceWidgets', '/').classes('text-subtitle1 no-underline')
            for path, title, _description in DEMO_ROUTES:
                ui.link(title, path).classes('no-underline')
            ui.space()
            ui.switch(
                'Dark',
                value=_APP_DARK_MODE,
                on_change=lambda e: on_theme_change(bool(e.value)),
            )
            ui.link('Docs', DOCS_URL, new_tab=True).classes('no-underline')

        with ui.column().classes('w-full flex-1 min-h-0 overflow-auto gap-0'):
            page = builder(embedded=True, dark_mode=_APP_DARK_MODE)
            page_holder['page'] = page


@ui.page('/')
def home_page() -> None:
    """Build the combined demo index."""
    ui.dark_mode(value=_APP_DARK_MODE)
    with ui.column().classes('w-full min-h-screen gap-0'):
        build_toolbar()
        with ui.column().classes('w-full gap-6 p-6'):
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


@ui.page('/plotly')
def plotly_page() -> None:
    """Build the PlotlyPlotWidget route."""
    build_demo_route(build_plotly_plot_demo_page)


@ui.page('/tree')
def tree_page() -> None:
    """Build the TreeWidget route."""
    build_demo_route(build_tree_demo_page)


def main() -> None:
    """Run the combined demo in a browser-friendly server configuration."""
    ui.run(
        title='NiceWidgets demos',
        host=os.getenv('NICEWIDGETS_DEMO_HOST', '127.0.0.1'),
        port=int(os.getenv('PORT', '8080')),
        reload=False,
        dark=_APP_DARK_MODE,
    )


if __name__ in {'__main__', '__mp_main__'}:
    main()
