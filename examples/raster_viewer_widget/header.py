"""Theme-aware NiceGUI header for the raster viewer demonstration."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum

from nicegui import events, ui

LOGGER = logging.getLogger(__name__)
ThemeChangeHandler = Callable[[str], Awaitable[str]]


class ViewerTheme(StrEnum):
    """Themes supported by the NiceGUI page and JavaScript viewer."""

    LIGHT = "light"
    DARK = "dark"


class DemoHeader:
    """Render the application header and own its light/dark theme selection."""

    def __init__(self, initial_theme: ViewerTheme = ViewerTheme.DARK) -> None:
        """Create the header with a compact binary theme switch.

        Args:
            initial_theme: Theme applied when the page is first rendered.
        """
        self._theme = initial_theme
        self._on_theme_change: ThemeChangeHandler | None = None
        self._dark_mode = ui.dark_mode(value=initial_theme is ViewerTheme.DARK)
        with ui.header().classes("items-center gap-3"):
            ui.label("JavaScript Raster Viewer").classes("text-h6 font-medium")
            ui.space()
            ui.icon("light_mode").props("size=sm").tooltip("Light theme")
            self._theme_switch = ui.switch(
                value=initial_theme is ViewerTheme.DARK,
                on_change=self._handle_theme_change,
            ).props('dense keep-color color=white aria-label="Application theme"')
            ui.icon("dark_mode").props("size=sm").tooltip("Dark theme")

    @property
    def theme(self) -> ViewerTheme:
        """Return the current header theme.

        Returns:
            Current light or dark theme.
        """
        return self._theme

    def bind_theme_change(self, handler: ThemeChangeHandler) -> None:
        """Bind one callback to future theme changes.

        Args:
            handler: Async callback accepting ``light`` or ``dark``.
        """
        self._on_theme_change = handler

    async def _handle_theme_change(
        self,
        event: events.ValueChangeEventArguments[bool | None],
    ) -> None:
        """Apply one slider change to NiceGUI and the child viewer.

        Args:
            event: Switch event whose value is true for dark.
        """
        self._theme = ViewerTheme.DARK if bool(event.value) else ViewerTheme.LIGHT
        self._dark_mode.set_value(self._theme is ViewerTheme.DARK)
        if self._on_theme_change is not None:
            applied = await self._on_theme_change(self._theme.value)
            LOGGER.info("Applied application theme: requested=%s applied=%s", self._theme, applied)
