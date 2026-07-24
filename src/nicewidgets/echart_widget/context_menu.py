"""Right-click context menu for the ECharts widget."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from nicewidgets.echart_widget.widget import EChartWidget

_CHECK_PREFIX = "✓ "


class EChartWidgetContextMenu:
    """Build the ECharts widget right-click context menu.

    The menu items call public widget setters only; the menu does not own any
    state. Check marks reflect the widget's :class:`EChartDisplayOptions` at
    the moment the menu is opened (the menu is rebuilt every right-click).

    Args:
        get_widget: Callable returning the widget that owns this menu.
    """

    def __init__(self, get_widget: Callable[[], EChartWidget]) -> None:
        self._get_widget = get_widget

    def build(self) -> None:
        """Populate the active NiceGUI context menu.

        Call this from inside ``with context_menu.clear():`` so items are
        inserted into the current menu instance.
        """
        widget = self._get_widget()
        options = widget.display_options

        ui.menu_item(
            self._toggle_label("Show Toolbar", options.show_toolbar),
            on_click=lambda: widget.set_toolbar_visible(not options.show_toolbar),
        )
        ui.menu_item(
            self._toggle_label("Hover Info", options.show_hover_info),
            on_click=lambda: widget.set_hover_info_visible(not options.show_hover_info),
        )
        ui.menu_item(
            self._toggle_label("Axis Labels", options.show_axis_labels),
            on_click=lambda: widget.set_axis_labels_visible(not options.show_axis_labels),
        )
        ui.menu_item(
            self._toggle_label("Horizontal Lines", options.show_horizontal_lines),
            on_click=lambda: widget.set_horizontal_lines_visible(
                not options.show_horizontal_lines
            ),
        )
        ui.menu_item(
            self._toggle_label("Vertical Lines", options.show_vertical_lines),
            on_click=lambda: widget.set_vertical_lines_visible(
                not options.show_vertical_lines
            ),
        )
        ui.separator()
        ui.menu_item("Copy To Clipboard", on_click=widget.copy_plot_to_clipboard)

    @staticmethod
    def _toggle_label(label: str, checked: bool) -> str:
        """Return a menu label with a check prefix when checked.

        Args:
            label: Base menu label.
            checked: Whether the item is currently enabled.

        Returns:
            Label string suitable for :func:`nicegui.ui.menu_item`.
        """
        return f'{_CHECK_PREFIX if checked else ""}{label}'
