"""Lazy section widget for deferring expensive UI rendering until expansion is opened.

Provides LazySection and LazySectionConfig classes for lazy-rendering UI content
inside a ui.expansion widget. Content is only built when the expansion is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from nicegui import ui


@dataclass
class LazySectionConfig:
    """Configuration for LazySection.

    Args:
        render_once: Render only the first time the section is opened.
        clear_on_close: Clear the section content when the section is closed.
            If clear_on_close=True and render_once=False, the section will rebuild on next open.
        show_spinner: Show a spinner while heavy compute is running.
    """
    render_once: bool = True
    clear_on_close: bool = False
    show_spinner: bool = True


class LazySection:
    """Lazy-render an expansion section.

    Purpose:
        Defer expensive work until the user opens a ui.expansion.

    Inputs:
        title: Expansion title.
        render_fn: Called on the UI thread to build widgets into the provided container.
        config: Controls render-once vs rebuild and optional clear-on-close.
        subtitle: Optional helper text shown at the top of the section.

    Output:
        Renders a ui.expansion whose inner content is constructed on-demand.
    """

    def __init__(
        self,
        title: str,
        *,
        render_fn: Callable[[ui.element], None],
        config: LazySectionConfig | None = None,
        subtitle: Optional[str] = None,
    ) -> None:
        self._title = title
        self._subtitle = subtitle
        self._render_fn = render_fn
        self._cfg = config or LazySectionConfig()

        self._rendered = False
        self._rendering = False

        # Expansion UI
        with ui.expansion(title, value=False).classes("w-full") as exp:
            self._expansion = exp

            with ui.column().classes("w-full gap-2"):
                if self._subtitle:
                    ui.label(self._subtitle).classes("text-sm text-gray-500")

                self._placeholder = ui.label("Open to load…").classes("text-sm text-gray-500")
                self._spinner = ui.spinner(size="md")
                self._spinner.visible = False

                self._content = ui.column().classes("w-full")

        # NiceGUI/Quasar: listen to open/close via model-value updates
        # e.args is the new bool (open state)
        self._expansion.on("update:model-value", self._on_model_value)

    async def _on_model_value(self, e) -> None:
        """Handle expansion open/close.

        e.args is expected to be a bool (open state).
        """
        opened = bool(getattr(e, "args", False))
        if opened:
            await self._on_open()
        else:
            self._on_close()

    async def _on_open(self) -> None:
        """Render content when opened (lazily)."""
        if self._cfg.render_once and self._rendered:
            return
        if self._rendering:
            return

        self._rendering = True
        self._placeholder.text = "Loading…"
        if self._cfg.show_spinner:
            self._spinner.visible = True

        # Always rebuild the content area when opening (unless render_once and already rendered)
        self._content.clear()

        try:
            # UI build on the UI thread
            self._render_fn(self._content)

            self._rendered = True
            self._placeholder.visible = False

        finally:
            self._rendering = False
            self._spinner.visible = False

    def _on_close(self) -> None:
        """Optionally clear content when closed to free memory."""
        if not self._cfg.clear_on_close:
            return
        if not self._rendered:
            return

        self._content.clear()
        self._placeholder.visible = True
        self._placeholder.text = "Open to load…"

        if not self._cfg.render_once:
            self._rendered = False
