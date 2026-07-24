"""Lazy NiceGUI expansion helper used by reusable widgets."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

from nicegui import ui


@dataclass(frozen=True, slots=True)
class LazySectionConfig:
    """Configuration for :class:`LazySection`.

    Args:
        render_once: Whether content should be rendered only the first time the
            expansion is opened.
        clear_on_close: Whether rendered content should be cleared when the
            expansion closes.
        show_spinner: Whether to show a small placeholder while rendering.
    """

    render_once: bool = True
    clear_on_close: bool = False
    show_spinner: bool = True


class LazySection:
    """Small wrapper that renders content when an expansion is opened.

    Args:
        title: Expansion title.
        subtitle: Optional subtitle shown inside the expansion before content.
        render_fn: Callback receiving the content container.
        config: Lazy rendering configuration.
    """

    def __init__(
        self,
        title: str,
        *,
        subtitle: str | None = None,
        render_fn: Callable[[ui.element], None],
        config: LazySectionConfig | None = None,
    ) -> None:
        self.title = title
        self.subtitle = subtitle
        self.render_fn = render_fn
        self.config = config if config is not None else LazySectionConfig()
        self.has_rendered = False
        self.expansion = ui.expansion(title).classes("w-full")
        with self.expansion:
            if subtitle:
                ui.label(subtitle).classes("text-sm text-gray-600")
            self.container = ui.column().classes("w-full")
        self.expansion.on("update:model-value", self._on_toggle)

    def _on_toggle(self, event: object) -> None:
        """Render or clear content when the expansion state changes.

        Args:
            event: NiceGUI event object.
        """
        value = bool(getattr(event, "args", False))
        if value:
            if self.config.render_once and self.has_rendered:
                return
            self.container.clear()
            with self.container:
                if self.config.show_spinner:
                    ui.spinner(size="sm")
            self.container.clear()
            self.render_fn(self.container)
            self.has_rendered = True
            return
        if self.config.clear_on_close:
            self.container.clear()
            self.has_rendered = False
