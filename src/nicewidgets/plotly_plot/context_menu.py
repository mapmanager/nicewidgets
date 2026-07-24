"""Right-click context menu for the reusable Plotly plot widget."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from nicewidgets.plotly_plot.widget import PlotlyPlotWidget

_CHECK_PREFIX = "✓ "


class PlotlyPlotContextMenu:
    """Build the Plotly plot widget right-click context menu.

    Args:
        get_widget: Callable returning the widget that owns this menu.
    """

    def __init__(self, get_widget: Callable[[], PlotlyPlotWidget]) -> None:
        self._get_widget = get_widget

    def build(self) -> None:
        """Populate the active NiceGUI context menu.

        This method should be called from inside ``with context_menu.clear():`` so
        menu items are inserted into the current menu instance.

        Returns:
            None.
        """
        widget = self._get_widget()
        options = widget.display_options

        for item in widget.series_menu_items:
            if item.separator_before:
                ui.separator()
            ui.menu_item(
                self._toggle_label(item.label, widget.is_series_visible(item.series_name)),
                on_click=lambda name=item.series_name: widget.toggle_series_visible(name),
            )

        if widget.series_menu_items:
            ui.separator()

        ui.menu_item(
            self._toggle_label("X Axis Labels", options.show_x_axis_labels),
            on_click=lambda: widget.set_x_axis_labels_visible(not options.show_x_axis_labels),
        )
        ui.menu_item(
            self._toggle_label("Y Axis Labels", options.show_y_axis_labels),
            on_click=lambda: widget.set_y_axis_labels_visible(not options.show_y_axis_labels),
        )
        ui.menu_item(
            self._toggle_label("Plotly Toolbar", options.show_plotly_toolbar),
            on_click=lambda: widget.set_plotly_toolbar_visible(not options.show_plotly_toolbar),
        )
        ui.menu_item(
            self._toggle_label("Hover Info", options.show_hover_info),
            on_click=lambda: widget.set_hover_info_visible(not options.show_hover_info),
        )
        ui.menu_item(
            self._toggle_label("Legend", options.show_legend),
            on_click=lambda: widget.set_legend_visible(not options.show_legend),
        )

        if widget.on_build_context_menu is not None:
            ui.separator()
            widget.on_build_context_menu(widget)

        ui.separator()
        ui.menu_item("Copy To Clipboard", on_click=widget.copy_plot_to_clipboard)

    @staticmethod
    def _toggle_label(label: str, checked: bool) -> str:
        """Return a menu label with a check prefix when checked.

        Args:
            label: Base menu label.
            checked: Whether the item is currently enabled.

        Returns:
            Label string suitable for ``ui.menu_item``.
        """
        return f"{_CHECK_PREFIX if checked else ''}{label}"
