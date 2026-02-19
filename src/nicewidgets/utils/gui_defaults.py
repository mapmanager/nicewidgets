"""Set up default classes and props for NiceGUI widgets.

This module provides a function to configure default styling for all NiceGUI
UI elements, including labels, buttons, checkboxes, selects, inputs, etc.
"""

from __future__ import annotations

from nicegui import ui

from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)


def setUpGuiDefaults(text_size: str = 'text-base'):
    """Set up default classes and props for all ui elements.
    
    Args:
        text_size: Tailwind CSS text size class (e.g., 'text-xs', 'text-sm', 
                   'text-base', 'text-lg'). Defaults to 'text-base'.
    """
    
    # logger.info('setting default_classes() and default_props()to specify style of all ui elements')
    
    # map tailwind to quasar size
    text_size_quasar = {
        "text-xs": "xs",
        "text-sm": "sm",
        "text-base": "md",
        "text-lg": "lg",
    }[text_size]

    logger.debug(f'using classes text_size:"{text_size}" text_size_quasar:{text_size_quasar}')

    ui.label.default_classes(f"{text_size} select-text")  #  select-text allows double-click selection
    ui.label.default_props("dense")
    #
    ui.button.default_classes(text_size)
    ui.button.default_props("dense")
    #
    ui.checkbox.default_classes(text_size)
    ui.checkbox.default_props(f"dense size={text_size_quasar}")
    # ui.checkbox.default_props("dense size=xs")
    # .props('size=xs')
    #
    ui.select.default_classes(text_size)
    ui.select.default_props("dense")
    #
    ui.input.default_classes(text_size)
    ui.input.default_props("dense")
    #
    ui.number.default_classes(text_size)
    ui.number.default_props("dense")
    #
    ui.expansion.default_classes(text_size)
    ui.expansion.default_props("dense")
    #
    ui.slider.default_classes(text_size)
    ui.slider.default_props("dense")
    #
    ui.linear_progress.default_classes(text_size)
    ui.linear_progress.default_props("dense")

    ui.menu.default_classes(text_size)
    ui.menu.default_props("dense")

    ui.menu_item.default_classes(text_size)
    ui.menu_item.default_props("dense")

    ui.radio.default_classes(text_size)
    ui.radio.default_props("dense")
