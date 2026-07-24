"""Right-click context menu for the Plotly raster viewer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from nicegui import ui

if TYPE_CHECKING:
    from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer

_CHECK_PREFIX = '✓ '


class PlotlyRasterViewerContextMenu:
    """Build the Plotly raster viewer right-click context menu.

    Args:
        get_viewer: Callable returning the viewer that owns this menu.
    """

    def __init__(self, get_viewer: Callable[[], PlotlyRasterViewer]) -> None:
        self._get_viewer = get_viewer

    def build(self) -> None:
        """Populate the active NiceGUI context menu.

        This method should be called from inside ``with context_menu.clear():`` so
        menu items are inserted into the current menu instance.
        """
        viewer = self._get_viewer()
        options = viewer.display_options

        ui.menu_item(
            self._toggle_label('ROIs', options.show_rois),
            on_click=lambda: viewer.set_roi_overlays_visible(not options.show_rois),
        )
        roi_labels_item = ui.menu_item(
            self._toggle_label('ROI Labels', options.show_roi_labels),
            on_click=lambda: viewer.set_roi_labels_visible(not options.show_roi_labels),
        )
        # ROI labels are meaningless while ROIs are hidden; keep the item shown
        # but disabled so it cannot be toggled until ROIs are visible again.
        roi_labels_item.set_enabled(options.show_rois)
        ui.menu_item(
            self._toggle_label('Traces', options.show_trace_overlays),
            on_click=lambda: viewer.set_trace_overlays_visible(not options.show_trace_overlays),
        )
        ui.menu_item(
            self._toggle_label('X Axis Labels', options.show_x_axis_labels),
            on_click=lambda: viewer.set_x_axis_labels_visible(not options.show_x_axis_labels),
        )
        ui.menu_item(
            self._toggle_label('Y Axis Labels', options.show_y_axis_labels),
            on_click=lambda: viewer.set_y_axis_labels_visible(not options.show_y_axis_labels),
        )
        ui.menu_item(
            self._toggle_label('Square Plot', options.square_plot),
            on_click=lambda: viewer.set_square_plot(not options.square_plot),
        )
        ui.menu_item(
            self._toggle_label('Plotly Toolbar', options.show_plotly_toolbar),
            on_click=lambda: viewer.set_plotly_toolbar_visible(not options.show_plotly_toolbar),
        )
        ui.menu_item(
            self._toggle_label('Hover Info', options.show_hover_info),
            on_click=lambda: viewer.set_hover_info_visible(not options.show_hover_info),
        )

        ui.separator()
        ui.menu_item('Copy To Clipboard', on_click=viewer.copy_plot_to_clipboard)

    @staticmethod
    def _toggle_label(label: str, checked: bool) -> str:
        """Return a menu label with a check prefix when checked.

        Args:
            label: Base menu label.
            checked: Whether the item is currently enabled.

        Returns:
            Label string suitable for ``ui.menu_item``.
        """
        return f'{_CHECK_PREFIX if checked else ""}{label}'
