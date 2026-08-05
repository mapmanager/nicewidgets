"""Shared compact NiceGUI defaults for the raster-viewer demonstration."""

from __future__ import annotations

from typing import Literal

from nicegui import ui

GuiTextSize = Literal["text-xs", "text-sm", "text-base", "text-lg"]


def set_up_gui_defaults(text_size: GuiTextSize = "text-xs") -> None:
    """Configure compact classes and properties before creating UI elements.

    Args:
        text_size: Tailwind text-size class applied to supported UI elements.

    Returns:
        None.
    """
    quasar_size = {
        "text-xs": "xs",
        "text-sm": "sm",
        "text-base": "md",
        "text-lg": "lg",
    }[text_size]

    ui.label.default_classes(f"{text_size} select-text")
    ui.label.default_props("dense")
    ui.button.default_classes(text_size)
    ui.button.default_props("dense")
    ui.checkbox.default_classes(text_size)
    ui.checkbox.default_props(f"dense size={quasar_size}")
    ui.select.default_classes(text_size)
    ui.select.default_props("dense")
    ui.input.default_classes(text_size)
    ui.input.default_props("dense")
    ui.number.default_classes(text_size)
    ui.number.default_props("dense")
    ui.expansion.default_classes(text_size)
    ui.expansion.default_props("dense")
    ui.slider.default_classes(text_size)
    ui.slider.default_props("dense")
    ui.linear_progress.default_classes(text_size)
    ui.linear_progress.default_props("dense")
    ui.menu.default_classes(text_size)
    ui.menu.default_props("dense")
    ui.menu_item.default_classes(text_size)
    ui.menu_item.default_props("dense")
    ui.radio.default_classes(text_size)
    ui.radio.default_props("dense")
