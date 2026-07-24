"""Reusable NiceGUI Plotly plotting widget."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from nicegui import core, ui

from nicewidgets.plotly_plot.context_menu import PlotlyPlotContextMenu
from nicewidgets.plotly_plot.context_menu_guards import pywebview_plotly_plot_context_menu_guard_js
from nicewidgets.plotly_plot.display_options import PlotlyPlotDisplayOptions
from nicewidgets.plotly_axis_layout import (
    PLOTLY_AXIS_LABEL_FONT_SIZE,
    any_axis_labels_visible,
    apply_axis_decorations,
    resolve_plot_layout_margins,
)
from nicewidgets.plotly_layout_margins import PlotlyLayoutMarginsProfile
from nicewidgets.plotly_plot.event_overlay import PlotlyEventOverlayApi
from nicewidgets.plotly_plot.models import (
    MeasurementChangeEvent,
    MeasurementLine,
    MeasurementPair,
    PlotlyAxisRange,
    PlotlyLineOrientation,
    PlotlyScatterData,
    PlotlySeriesMenuItem,
    PlotlyTraceData,
    PlotlyYAxisSide,
    _normalize_y_axis_side,
)
from nicewidgets.raster_viewer.frontend.plotly_clipboard import (
    copy_plotly_png_to_browser_clipboard,
    get_plotly_png_bytes,
)
from nicewidgets.utils.clipboard import copy_png_bytes_to_native_clipboard
from nicewidgets.utils.desktop import is_pywebview_desktop
from nicewidgets.plotly_theme import (
    PlotlyThemeName,
    apply_plotly_theme_to_layout,
    normalize_plotly_theme,
    theme_for_name,
)
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

OnPlotlyXRangeChanged = Callable[[float | None, float | None], None]
OnSeriesVisibilityChanged = Callable[[str, bool], None]
OnPlotlyXRangeSelected = Callable[[float, float], None]
OnMeasurementChanged = Callable[[MeasurementChangeEvent], None]

_X_RANGE_ECHO_EPS = 1e-9
_SELF_RELAYOUT_TTL_SEC = 0.250


def _x_range_equal(
    a: tuple[float | None, float | None],
    b: tuple[float | None, float | None],
) -> bool:
    """Compare two ``(x_min, x_max)`` pairs with float tolerance and ``None`` support."""
    for av, bv in zip(a, b, strict=True):
        if av is None or bv is None:
            if av is not bv:
                return False
            continue
        if not (math.isfinite(av) and math.isfinite(bv)):
            return False
        if abs(av - bv) > _X_RANGE_ECHO_EPS:
            return False
    return True


def _relayout_has_axis_range(args: dict[str, object]) -> bool:
    """Return whether ``args`` carries any Plotly axis range keys."""
    return any(key.startswith("xaxis.range") or key.startswith("yaxis.range") for key in args)


def _relayout_has_bracket_axis_range(args: dict[str, object]) -> bool:
    """Return whether ``args`` carries bracket-style axis range keys from user gestures."""
    return any(
        ("[0]" in key or "[1]" in key)
        and (key.startswith("xaxis.range") or key.startswith("yaxis.range"))
        for key in args
    )


def _is_normalized_only_relayout(args: dict[str, object]) -> bool:
    """Return whether ``args`` is a normalized list-key relayout without bracket keys."""
    has_list_range = "xaxis.range" in args or "yaxis.range" in args
    has_bracket_range = any("[0]" in key or "[1]" in key for key in args)
    return has_list_range and not has_bracket_range


def extract_rect_selection_x_range_from_relayout(
    args: dict[str, object],
) -> tuple[float | None, float | None]:
    """Parse Plotly relayout args for box-select ``x0``/``x1``.

    Plotly delivers box-select as a relayout payload. Keys vary (flat ``selections[0].*``,
    other indexed entries, or a ``selections`` list). Only the x-range is returned.

    Args:
        args: Plotly relayout event payload.

    Returns:
        Parsed ``(x0, x1)`` or ``(None, None)`` when not found.
    """
    if not args:
        return None, None
    k0, k1 = "selections[0].x0", "selections[0].x1"
    if k0 in args and k1 in args:
        try:
            return float(args[k0]), float(args[k1])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    for key in list(args.keys()):
        if ".x0" in key and key.startswith("selections[") and key.endswith(".x0"):
            prefix = key[:-3]
            k1_alt = f"{prefix}.x1"
            if k1_alt in args:
                try:
                    return float(args[key]), float(args[k1_alt])  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    pass
    raw = args.get("selections")
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, dict):
                x0 = item.get("x0")
                x1 = item.get("x1")
                if x0 is not None and x1 is not None:
                    try:
                        return float(x0), float(x1)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        pass
    return None, None


_SeriesKind = Literal["trace", "scatter"]
_MeasurementKind = Literal["line", "pair"]


@dataclass(slots=True)
class _SeriesRef:
    """Internal mapping from a public series name to a Plotly trace index."""

    name: str
    kind: _SeriesKind


@dataclass(slots=True)
class _ShapeRef:
    """Internal mapping from a Plotly shape index to a measurement object."""

    name: str
    kind: _MeasurementKind
    line_number: int


def _normalize_orientation(orientation: str) -> PlotlyLineOrientation:
    """Normalize supported orientation aliases.

    Args:
        orientation: Orientation string. Supported values are ``horizontal``,
            ``vertical``, ``h``, and ``v``.

    Returns:
        Normalized orientation literal.

    Raises:
        ValueError: If the orientation is not supported.
    """
    value = str(orientation).strip().lower()
    if value in {"horizontal", "h"}:
        return "horizontal"
    if value in {"vertical", "v"}:
        return "vertical"
    raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation!r}")


def _validate_unique_name(name: str, existing: object | None, *, label: str) -> str:
    """Validate a non-empty unique public name.

    Args:
        name: Candidate name.
        existing: Existing object for this name, if any.
        label: Human-readable label for validation errors.

    Returns:
        Stripped name.

    Raises:
        ValueError: If the name is empty or already exists.
    """
    clean = str(name).strip()
    if not clean:
        raise ValueError(f"{label} name must not be empty")
    if existing is not None:
        raise ValueError(f"{label} {clean!r} already exists")
    return clean


_PLOTLY_PLOT_LEGEND: dict[str, Any] = {
    "orientation": "h",
    "xanchor": "center",
    "x": 0.5,
    "yanchor": "top",
    "y": -0.15,
}


def build_plotly_figure_dict(
    *,
    data: list[dict[str, Any]] | None = None,
    x_label: str = "x",
    y_label: str = "y",
    x_range: PlotlyAxisRange | None = None,
    shapes: list[dict[str, Any]] | None = None,
    theme: PlotlyThemeName = "light",
    show_x_axis_labels: bool = False,
    show_y_axis_labels: bool = False,
    show_legend: bool = True,
    show_plotly_toolbar: bool = False,
) -> dict[str, Any]:
    """Build a NiceGUI-compatible Plotly figure dictionary.

    Args:
        data: Plotly trace dictionaries.
        x_label: X-axis label.
        y_label: Y-axis label.
        x_range: Optional explicit x-axis range.
        shapes: Optional Plotly layout shapes.
        theme: Plotly light/dark layout theme name.
        show_x_axis_labels: Whether x-axis decorations are visible.
        show_y_axis_labels: Whether primary y-axis decorations are visible.
        show_legend: Whether the bottom horizontal legend is visible.
        show_plotly_toolbar: Whether Plotly's modebar is visible.

    Returns:
        Dictionary with Plotly ``data``, ``layout``, and ``config`` keys.
    """
    range_model = x_range or PlotlyAxisRange()
    xaxis: dict[str, Any] = {}
    apply_axis_decorations(
        xaxis,
        label_text=x_label,
        visible=bool(show_x_axis_labels),
    )
    if range_model.x_min is not None and range_model.x_max is not None:
        xaxis["range"] = [range_model.x_min, range_model.x_max]
        xaxis["autorange"] = False
    else:
        xaxis["autorange"] = True

    yaxis: dict[str, Any] = {"autorange": True}
    apply_axis_decorations(
        yaxis,
        label_text=y_label,
        visible=bool(show_y_axis_labels),
    )

    margin = resolve_plot_layout_margins(
        show_axis_labels=any_axis_labels_visible(
            show_x_axis_labels=show_x_axis_labels,
            show_y_axis_labels=show_y_axis_labels,
        ),
        show_legend=bool(show_legend),
    )

    layout: dict[str, Any] = {
        "xaxis": xaxis,
        "yaxis": yaxis,
        "shapes": list(shapes or []),
        "dragmode": "zoom",
        "margin": margin,
        "showlegend": bool(show_legend),
        "legend": dict(_PLOTLY_PLOT_LEGEND),
        "uirevision": "nicewidgets-plotly-plot",
    }
    apply_plotly_theme_to_layout(layout, normalize_plotly_theme(theme))
    return {
        "data": list(data or []),
        "layout": layout,
        "config": {
            "editable": True,
            "scrollZoom": True,
            "displaylogo": False,
            "responsive": True,
            "displayModeBar": bool(show_plotly_toolbar),
            "edits": {
                "shapePosition": True,
                "titleText": False,
                "axisTitleText": False,
                "legendText": False,
                "legendPosition": False,
            },
        },
    }


class PlotlyPlotWidget:
    """Interactive Plotly plotting widget for NiceGUI.

    This widget provides a reusable plotting interface for scientific traces,
    sparse marker overlays, editable measurement lines, and x-axis range
    synchronization. It intentionally hides Plotly layout-shape details from
    parent NiceGUI applications.
    """

    def __init__(
        self,
        *,
        x_label: str = "x",
        y_label: str = "y",
        y2_label: str = "",
        display_options: PlotlyPlotDisplayOptions | None = None,
        on_x_range_changed: OnPlotlyXRangeChanged | None = None,
        on_x_range_selected: OnPlotlyXRangeSelected | None = None,
        on_measurement_changed: OnMeasurementChanged | None = None,
        on_series_visibility_changed: OnSeriesVisibilityChanged | None = None,
        on_build_context_menu: Callable[[PlotlyPlotWidget], None] | None = None,
        layout_margins_profile: PlotlyLayoutMarginsProfile | None = None,
    ) -> None:
        """Create an empty Plotly widget.

        Args:
            x_label: X-axis label.
            y_label: Primary left y-axis label.
            y2_label: Secondary right y-axis label used when a visible right-axis
                trace or scatter is present.
            display_options: Initial display options (theme, legend, axis-label,
                toolbar, and hover visibility). Defaults to
                :class:`PlotlyPlotDisplayOptions` defaults. The widget owns a
                private copy; later context-menu toggles mutate the copy.
            on_x_range_changed: Optional callback invoked after the user changes
                the x-axis range by zooming, panning, or autoranging. ``(None,
                None)`` means Plotly returned to autorange.
            on_x_range_selected: Optional callback invoked once after the user
                completes a box-select while ``begin_select_x_range()`` is armed.
            on_measurement_changed: Optional callback invoked after the user
                drags a measurement line.
            on_series_visibility_changed: Optional callback invoked after a
                context-menu series visibility toggle.
            on_build_context_menu: Optional callback invoked while rebuilding the
                right-click context menu, after built-in display toggles and
                before Copy To Clipboard. Callers may add arbitrary
                ``ui.menu_item`` / separator entries (same pattern as
                ``TableWidget`` / ``TreeWidget``).
            layout_margins_profile: Optional fixed margin profile for aligned
                multi-plot stacks.
        """
        self._x_label = str(x_label)
        self._y_label = str(y_label)
        self._y2_label = str(y2_label)
        self._layout_margins_profile = layout_margins_profile
        self._display_options = replace(display_options or PlotlyPlotDisplayOptions())
        self._theme = normalize_plotly_theme(self._display_options.theme)
        self._display_options.theme = self._theme
        self._placeholder_text: str | None = None
        self._on_x_range_changed = on_x_range_changed
        self._on_x_range_selected = on_x_range_selected
        self._on_measurement_changed = on_measurement_changed
        self._on_series_visibility_changed = on_series_visibility_changed
        self._on_build_context_menu = on_build_context_menu
        self._x_range = PlotlyAxisRange()
        self._series_menu_items: list[PlotlySeriesMenuItem] = []
        self._series_visibility: dict[str, bool] = {}
        self._figure = build_plotly_figure_dict(
            x_label=self._x_label,
            y_label=self._y_label,
            x_range=self._x_range,
            theme=self._theme,
            show_x_axis_labels=self._display_options.show_x_axis_labels,
            show_y_axis_labels=self._display_options.show_y_axis_labels,
            show_legend=self._display_options.show_legend,
            show_plotly_toolbar=self._display_options.show_plotly_toolbar,
        )
        self._series_order: list[_SeriesRef] = []
        self._traces: dict[str, PlotlyTraceData] = {}
        self._scatters: dict[str, PlotlyScatterData] = {}
        self._measurements: dict[str, MeasurementLine | MeasurementPair] = {}
        self._shape_refs: list[_ShapeRef] = []
        self._measurement_callbacks: dict[str, OnMeasurementChanged] = {}
        self._last_applied_x_range: tuple[float | None, float | None] | None = None
        self._ignore_relayout = False
        self._x_range_selection_armed = False
        self._pending_self_relayouts: list[dict[str, object]] = []
        self._ctx_menu: ui.context_menu | None = None
        self._context_menu_builder: PlotlyPlotContextMenu | None = None
        self.events = PlotlyEventOverlayApi(self)
        if self._layout_margins_profile is not None:
            self._sync_margins_to_plotly_dict()
            self._sync_axis_stabilization_to_plotly_dict()

        with ui.element("div").classes(
            "relative w-full h-full min-h-0 nw-plotly-plot"
        ) as self.container:
            self._plot_element = ui.plotly(self._figure).classes("w-full h-full min-h-0")
            with ui.element("div").classes(
                "absolute inset-0 flex items-center justify-center pointer-events-none px-4"
            ) as self._placeholder_container:
                self._placeholder_label = ui.label("").classes("text-sm opacity-70 text-center")
        self._placeholder_container.set_visibility(False)
        self._ensure_measurement_drag_css()
        self._plot_element.on("plotly_relayout", self._on_plotly_relayout)
        self._plot_element.on("plotly_doubleclick", self._on_plotly_doubleclick)
        self._ctx_menu = ui.context_menu()
        self._context_menu_builder = PlotlyPlotContextMenu(get_widget=lambda: self)
        self._plot_element.on("contextmenu", self._on_context_menu_event)
        if is_pywebview_desktop():
            ui.timer(0.05, self._install_pywebview_context_menu_guards, once=True)

    @staticmethod
    def _ensure_measurement_drag_css() -> None:
        """Disable Plotly shape vertex handles so line bodies drag as a unit.

        Plotly's shape editor exposes endpoint circles. Dragging a circle moves
        one endpoint and looks like a broken diagonal / “first point” drag.
        Measurement H/V lines should translate as one axis-aligned segment.
        Source: Plotly community guidance for ``config.edits.shapePosition``
        (disable pointer events on shape vertex circles).

        Returns:
            None.
        """
        ui.add_head_html(
            """
<style id="nw-plotly-measurement-drag-css">
/* Prefer dragging the line body, not endpoint vertex circles. */
.nw-plotly-plot .js-plotly-plot .draglayer circle,
.nw-plotly-plot .js-plotly-plot g.draglayer circle {
  pointer-events: none !important;
}
</style>
""",
            shared=True,
        )

    @property
    def display_options(self) -> PlotlyPlotDisplayOptions:
        """Return mutable display options used by context-menu actions."""
        return self._display_options

    @property
    def placeholder_text(self) -> str | None:
        """Return the current centered placeholder message, if any."""
        return self._placeholder_text

    def set_placeholder_text(self, message: str | None) -> None:
        """Show or hide centered placeholder text over the plot area.

        Args:
            message: Human-readable empty-state text, or ``None`` to hide the
                placeholder overlay.

        Returns:
            None.
        """
        clean = str(message).strip() if message is not None else ""
        if not clean:
            self._placeholder_text = None
            self._placeholder_container.set_visibility(False)
            return
        self._placeholder_text = clean
        self._placeholder_label.text = clean
        self._placeholder_container.set_visibility(True)

    @property
    def series_menu_items(self) -> tuple[PlotlySeriesMenuItem, ...]:
        """Return registered trace/scatter context-menu items."""
        return tuple(self._series_menu_items)

    @property
    def on_build_context_menu(self) -> Callable[[PlotlyPlotWidget], None] | None:
        """Return optional callback that adds custom context-menu items."""
        return self._on_build_context_menu

    def set_on_build_context_menu(
        self,
        callback: Callable[[PlotlyPlotWidget], None] | None,
    ) -> None:
        """Set or clear the custom context-menu build callback.

        Args:
            callback: Invoked while rebuilding the right-click menu, or ``None``.

        Returns:
            None.
        """
        self._on_build_context_menu = callback

    def register_series_menu_items(self, items: Sequence[PlotlySeriesMenuItem]) -> None:
        """Register trace/scatter items shown in the right-click context menu.

        Existing visibility choices are preserved for series names that were
        registered previously in this widget instance.

        Args:
            items: Menu item definitions keyed by stable series names.

        Returns:
            None.
        """
        self._series_menu_items = list(items)
        for item in items:
            if item.series_name not in self._series_visibility:
                self._series_visibility[item.series_name] = bool(item.default_visible)

    def is_series_visible(self, series_name: str) -> bool:
        """Return whether one registered or loaded series is visible.

        Args:
            series_name: Stable trace or scatter overlay name.

        Returns:
            True when the series should render in the plot.
        """
        clean = str(series_name).strip()
        if clean in self._series_visibility:
            return bool(self._series_visibility[clean])
        return True

    def set_series_visible(self, series_name: str, visible: bool) -> None:
        """Set visibility for one loaded trace or scatter overlay.

        Args:
            series_name: Existing trace or scatter overlay name.
            visible: Whether the series should be visible.

        Raises:
            KeyError: If the series does not exist in the current figure.
        """
        clean = str(series_name).strip()
        self._series_visibility[clean] = bool(visible)
        if clean in self._traces:
            current = self._traces[clean]
            data = PlotlyTraceData.from_sequences(
                name=clean,
                x=current.x,
                y=current.y,
                visible=bool(visible),
                y_axis=current.y_axis,
                line_color=current.line_color,
                line_dash=current.line_dash,
            )
            self._traces[clean] = data
            index = self._series_index(clean, "trace")
            trace = self._trace_to_plotly(data)
            self._figure["data"][index] = trace
            self._restyle_plotly_trace(index, trace)
            self._refresh_yaxis2_layout()
            self._pin_x_axis_after_series_update()
            return
        if clean in self._scatters:
            current = self._scatters[clean]
            data = PlotlyScatterData.from_sequences(
                name=clean,
                x=current.x,
                y=current.y,
                visible=bool(visible),
                y_axis=current.y_axis,
            )
            self._scatters[clean] = data
            index = self._series_index(clean, "scatter")
            trace = self._scatter_to_plotly(data)
            self._figure["data"][index] = trace
            self._restyle_plotly_trace(index, trace)
            self._refresh_yaxis2_layout()
            self._pin_x_axis_after_series_update()
            return
        raise KeyError(f"series {clean!r} does not exist")

    def set_series_visible_state(self, series_name: str, visible: bool) -> None:
        """Set desired visibility for a series that may not be loaded yet.

        Unlike :meth:`set_series_visible`, this never raises for an unknown
        series. When the series is already loaded it restyles immediately;
        otherwise the visibility is stored and applied the next time the series
        is added. This supports restoring visibility before plot data exists
        (for example on reconnect hydrate).

        Args:
            series_name: Trace or scatter overlay name.
            visible: Whether the series should be visible.

        Returns:
            None.
        """
        clean = str(series_name).strip()
        if clean in self._traces or clean in self._scatters:
            self.set_series_visible(clean, visible)
            return
        self._series_visibility[clean] = bool(visible)

    def set_y2_label(self, label: str) -> None:
        """Set the secondary right y-axis title text.

        Decorations appear only when y-axis labels are enabled and at least one
        right-axis trace or scatter is visible.

        Args:
            label: Y2 axis title, or ``""`` to clear.

        Returns:
            None.
        """
        self._y2_label = str(label)
        if self._has_yaxis2():
            self._refresh_yaxis2_layout()

    def set_x_label(self, label: str) -> None:
        """Set the primary x-axis title text.

        Args:
            label: X-axis title, or ``""`` to clear.

        Returns:
            None.
        """
        self._set_primary_axis_label(axis_name="xaxis", label=str(label), attr="_x_label")

    def set_y_label(self, label: str) -> None:
        """Set the primary left y-axis title text.

        Args:
            label: Y-axis title, or ``""`` to clear.

        Returns:
            None.
        """
        self._set_primary_axis_label(axis_name="yaxis", label=str(label), attr="_y_label")

    def _set_primary_axis_label(self, *, axis_name: str, label: str, attr: str) -> None:
        """Update one primary axis title in memory and optionally relayout."""
        setattr(self, attr, label)
        layout = self._figure.setdefault("layout", {})
        axis = layout.setdefault(axis_name, {})
        if not isinstance(axis, dict):
            axis = {}
            layout[axis_name] = axis
        title = axis.setdefault("title", {})
        if not isinstance(title, dict):
            title = {}
            axis["title"] = title
        visible = (
            bool(self._display_options.show_x_axis_labels)
            if axis_name == "xaxis"
            else bool(self._display_options.show_y_axis_labels)
        )
        title["text"] = label if visible else ""
        if visible:
            self._relayout({f"{axis_name}.title.text": label})

    def _any_axis_labels_visible(self) -> bool:
        """Return whether any axis decorations are visible for margin layout."""
        return any_axis_labels_visible(
            show_x_axis_labels=self._display_options.show_x_axis_labels,
            show_y_axis_labels=self._display_options.show_y_axis_labels,
        )

    def toggle_series_visible(self, series_name: str) -> bool:
        """Toggle visibility for one registered trace or scatter overlay.

        Args:
            series_name: Stable trace or scatter overlay name.

        Returns:
            Visibility after the toggle.

        Raises:
            KeyError: If ``series_name`` is not a registered menu item.
        """
        clean = str(series_name).strip()
        if not any(item.series_name == clean for item in self._series_menu_items):
            raise KeyError(f"series {clean!r} is not registered in the context menu")
        new_visible = not self.is_series_visible(clean)
        if clean in self._traces or clean in self._scatters:
            self.set_series_visible(clean, new_visible)
        else:
            self._series_visibility[clean] = new_visible
        if self._on_series_visibility_changed is not None:
            self._on_series_visibility_changed(clean, new_visible)
        return new_visible

    def set_x_axis_labels_visible(self, visible: bool) -> None:
        """Show or hide x-axis title text, ticks, lines, and grid lines.

        Args:
            visible: Whether x-axis decorations should be visible.

        Returns:
            None.
        """
        self._display_options.show_x_axis_labels = bool(visible)
        self._sync_axis_labels_to_plotly_dict()
        self._sync_margins_to_plotly_dict()
        self._sync_axis_stabilization_to_plotly_dict()
        self._relayout_axis_labels_and_margins()

    def set_y_axis_labels_visible(self, visible: bool) -> None:
        """Show or hide left and right y-axis title text, ticks, lines, and grid lines.

        Args:
            visible: Whether y-axis decorations should be visible.

        Returns:
            None.
        """
        self._display_options.show_y_axis_labels = bool(visible)
        self._sync_axis_labels_to_plotly_dict()
        self._sync_margins_to_plotly_dict()
        self._sync_axis_stabilization_to_plotly_dict()
        self._relayout_axis_labels_and_margins()

    def set_plotly_toolbar_visible(self, visible: bool) -> None:
        """Set Plotly modebar visibility.

        Args:
            visible: Whether Plotly's modebar should be visible.

        Returns:
            None.
        """
        self._display_options.show_plotly_toolbar = bool(visible)
        self._sync_plotly_config_to_plotly_dict()
        self._react_plotly_config()

    def set_hover_info_visible(self, visible: bool) -> None:
        """Set Plotly hover-info visibility for all plot traces.

        Args:
            visible: Whether hover info should be visible.

        Returns:
            None.
        """
        self._display_options.show_hover_info = bool(visible)
        self._sync_hover_info_to_plotly_dict()
        self._restyle_hover_info()

    def set_legend_visible(self, visible: bool) -> None:
        """Show or hide the Plotly legend.

        When shown, the legend uses the widget's bottom horizontal layout
        (``orientation='h'`` centered below the plot).

        Args:
            visible: Whether the legend should be visible.

        Returns:
            None.
        """
        self._display_options.show_legend = bool(visible)
        self._sync_legend_to_plotly_dict()
        self._sync_margins_to_plotly_dict()
        self._relayout_legend()

    async def copy_plot_to_clipboard(self) -> None:
        """Copy the current Plotly plot image to the active clipboard.

        Native desktop mode uses ``pyperclipimg``. Browser mode uses the
        Clipboard API with a Plotly PNG export.

        Returns:
            None.
        """
        try:
            if is_pywebview_desktop():
                png_bytes = await get_plotly_png_bytes(self._plot_element)
                copy_png_bytes_to_native_clipboard(png_bytes)
            else:
                await copy_plotly_png_to_browser_clipboard(self._plot_element)
            ui.notify("Plot copied to clipboard.", type="positive")
        except Exception as exc:
            logger.exception("Failed to copy Plotly plot to clipboard.")
            ui.notify(f"Copy failed: {exc}", type="negative")

    def _on_context_menu_event(self, _event: Any) -> None:
        """Rebuild and open the Plotly plot context menu."""
        if self._ctx_menu is None or self._context_menu_builder is None:
            return
        with self._ctx_menu.clear():
            self._context_menu_builder.build()
        self._ctx_menu.open()

    def _install_pywebview_context_menu_guards(self) -> None:
        """Install desktop-only capture listeners so secondary taps open the menu."""
        js = pywebview_plotly_plot_context_menu_guard_js(plot_id=self._plot_element.id)
        try:
            self._plot_element.client.run_javascript(js, timeout=2.0)
        except RuntimeError:
            logger.debug("Could not install pywebview context-menu guards; client unavailable.")

    @property
    def figure(self) -> dict[str, Any]:
        """Return the current Plotly figure dictionary."""
        return self._figure

    def add_trace(
        self,
        *,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        visible: bool = True,
        y_axis: PlotlyYAxisSide = "left",
        line_color: str | None = None,
        line_dash: str | None = None,
    ) -> None:
        """Add a named continuous ``scattergl`` line trace.

        Args:
            name: Stable caller-defined trace name.
            x: X-axis values.
            y: Y-axis values.
            visible: Whether the trace should be visible.
            y_axis: Primary ``y`` axis (``"left"``) or overlaid ``y2`` axis
                (``"right"``). Right-axis traces create ``layout.yaxis2``.
            line_color: Optional Plotly line color.
            line_dash: Optional Plotly line dash.

        Raises:
            ValueError: If the name already exists or data are invalid.
        """
        clean = _validate_unique_name(name, self._traces.get(str(name).strip()), label="trace")
        axis = _normalize_y_axis_side(y_axis)
        data = PlotlyTraceData.from_sequences(
            name=clean,
            x=x,
            y=y,
            visible=visible,
            y_axis=axis,
            line_color=line_color,
            line_dash=line_dash,
        )
        self._traces[clean] = data
        self._series_order.append(_SeriesRef(name=clean, kind="trace"))
        self._sync_yaxis2_from_series()
        trace = self._trace_to_plotly(data)
        self._figure["data"].append(trace)
        self._add_plotly_trace(trace)

    def update_trace(
        self,
        *,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        visible: bool | None = None,
    ) -> None:
        """Replace data for an existing named continuous trace.

        Args:
            name: Existing trace name.
            x: Replacement X-axis values.
            y: Replacement Y-axis values.
            visible: Optional replacement visibility. When ``None``, the
                existing visibility is preserved.

        Raises:
            KeyError: If the trace does not exist.
            ValueError: If replacement data are invalid.
        """
        clean = str(name).strip()
        current = self._traces.get(clean)
        if current is None:
            raise KeyError(f"trace {clean!r} does not exist")
        data = PlotlyTraceData.from_sequences(
            name=clean,
            x=x,
            y=y,
            visible=current.visible if visible is None else visible,
            y_axis=current.y_axis,
            line_color=current.line_color,
            line_dash=current.line_dash,
        )
        self._traces[clean] = data
        index = self._series_index(clean, "trace")
        trace = self._trace_to_plotly(data)
        self._figure["data"][index] = trace
        self._restyle_plotly_trace(index, trace)

    def remove_trace(self, name: str) -> None:
        """Remove a named continuous trace.

        Args:
            name: Existing trace name.

        Raises:
            KeyError: If the trace does not exist.
        """
        clean = str(name).strip()
        index = self._series_index(clean, "trace")
        self._traces.pop(clean)
        self._series_order.pop(index)
        self._figure["data"].pop(index)
        self._delete_plotly_trace(index)
        self._sync_yaxis2_from_series()

    def clear_traces(self) -> None:
        """Remove all continuous traces while preserving scatter overlays."""
        for name in list(self._traces):
            self.remove_trace(name)

    def plot_scatter(
        self,
        *,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        visible: bool = True,
        y_axis: PlotlyYAxisSide = "left",
    ) -> None:
        """Add a named sparse ``scattergl`` marker overlay.

        Args:
            name: Stable caller-defined scatter overlay name.
            x: X-axis values.
            y: Y-axis values.
            visible: Whether the scatter overlay should be visible.
            y_axis: Primary ``y`` axis (``"left"``) or overlaid ``y2`` axis
                (``"right"``). Right-axis scatters create ``layout.yaxis2``.

        Raises:
            ValueError: If the name already exists or data are invalid.
        """
        clean = _validate_unique_name(
            name,
            self._scatters.get(str(name).strip()),
            label="scatter",
        )
        axis = _normalize_y_axis_side(y_axis)
        data = PlotlyScatterData.from_sequences(
            name=clean, x=x, y=y, visible=visible, y_axis=axis
        )
        self._scatters[clean] = data
        self._series_order.append(_SeriesRef(name=clean, kind="scatter"))
        self._sync_yaxis2_from_series()
        trace = self._scatter_to_plotly(data)
        self._figure["data"].append(trace)
        self._add_plotly_trace(trace)

    def update_scatter(
        self,
        *,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        visible: bool | None = None,
    ) -> None:
        """Replace data for an existing named scatter overlay.

        Args:
            name: Existing scatter overlay name.
            x: Replacement X-axis values.
            y: Replacement Y-axis values.
            visible: Optional replacement visibility. When ``None``, the
                existing visibility is preserved.

        Raises:
            KeyError: If the scatter overlay does not exist.
            ValueError: If replacement data are invalid.
        """
        clean = str(name).strip()
        current = self._scatters.get(clean)
        if current is None:
            raise KeyError(f"scatter {clean!r} does not exist")
        data = PlotlyScatterData.from_sequences(
            name=clean,
            x=x,
            y=y,
            visible=current.visible if visible is None else visible,
            y_axis=current.y_axis,
        )
        self._scatters[clean] = data
        index = self._series_index(clean, "scatter")
        trace = self._scatter_to_plotly(data)
        self._figure["data"][index] = trace
        self._restyle_plotly_trace(index, trace)

    def remove_scatter(self, name: str) -> None:
        """Remove a named scatter overlay.

        Args:
            name: Existing scatter overlay name.

        Raises:
            KeyError: If the scatter overlay does not exist.
        """
        clean = str(name).strip()
        index = self._series_index(clean, "scatter")
        self._scatters.pop(clean)
        self._series_order.pop(index)
        self._figure["data"].pop(index)
        self._delete_plotly_trace(index)
        self._sync_yaxis2_from_series()

    def clear_scatters(self) -> None:
        """Remove all scatter overlays while preserving continuous traces."""
        for name in list(self._scatters):
            self.remove_scatter(name)

    def set_series(
        self,
        *,
        traces: Sequence[PlotlyTraceData] = (),
        scatters: Sequence[PlotlyScatterData] = (),
    ) -> None:
        """Replace all continuous traces and scatter overlays in one browser update.

        Measurement lines and layout shapes are preserved. Existing incremental
        ``add_trace`` / ``plot_scatter`` callers remain available; prefer this
        method when rebuilding the full plot contents at once.

        Args:
            traces: Replacement continuous traces.
            scatters: Replacement scatter overlays.

        Returns:
            None.
        """
        self._traces = {}
        self._scatters = {}
        self._series_order = []
        plotly_data: list[dict[str, Any]] = []
        for data in traces:
            visible = self.is_series_visible(data.name)
            stored = PlotlyTraceData(
                name=data.name,
                x=data.x,
                y=data.y,
                visible=visible,
                y_axis=data.y_axis,
                line_color=data.line_color,
                line_dash=data.line_dash,
            )
            self._traces[stored.name] = stored
            self._series_order.append(_SeriesRef(name=stored.name, kind="trace"))
            plotly_data.append(self._trace_to_plotly(stored))
        for data in scatters:
            visible = self.is_series_visible(data.name)
            stored = PlotlyScatterData(
                name=data.name,
                x=data.x,
                y=data.y,
                visible=visible,
                y_axis=data.y_axis,
            )
            self._scatters[stored.name] = stored
            self._series_order.append(_SeriesRef(name=stored.name, kind="scatter"))
            plotly_data.append(self._scatter_to_plotly(stored))
        self._figure["data"] = plotly_data
        self._sync_hover_info_to_plotly_dict()
        self._sync_yaxis2_from_series()
        self._push_series_data()
        self._pin_x_axis_after_series_update()
        if plotly_data:
            self.set_placeholder_text(None)

    @staticmethod
    def _finite_x_values(values: Sequence[float]) -> list[float]:
        """Return finite x samples from one trace sequence."""
        return [float(value) for value in values if math.isfinite(value)]

    def _derive_x_range_from_visible_line_traces(self) -> tuple[float, float] | None:
        """Return the x extent of visible continuous line traces.

        Scatter overlays are excluded so marker padding does not expand the
        displayed x-axis when the logical range is automatic.
        """
        xs: list[float] = []
        for trace in self._traces.values():
            if not trace.visible:
                continue
            xs.extend(self._finite_x_values(trace.x))
        if not xs:
            return None
        return min(xs), max(xs)

    def _push_x_axis_range_to_browser(self, x_min: float, x_max: float) -> None:
        """Apply x-axis limits to the local figure dict and browser."""
        xaxis = self._figure["layout"].setdefault("xaxis", {})
        xaxis["range"] = [float(x_min), float(x_max)]
        xaxis["autorange"] = False
        self._relayout({"xaxis.range": [float(x_min), float(x_max)], "xaxis.autorange": False})

    def _pin_x_axis_after_series_update(self) -> None:
        """Pin x-axis limits after trace replacement.

        When the logical range is automatic, derive limits from visible line
        traces only. Scatter marker traces otherwise expand autorange padding.
        """
        x_min, x_max = self._x_range.x_min, self._x_range.x_max
        if x_min is None or x_max is None:
            derived = self._derive_x_range_from_visible_line_traces()
            if derived is None:
                return
            x_min, x_max = derived
        self._push_x_axis_range_to_browser(x_min, x_max)

    def set_theme(self, theme: PlotlyThemeName) -> None:
        """Set the Plotly light/dark layout theme.

        Args:
            theme: Theme name, either ``'light'`` or ``'dark'``.

        Returns:
            None.
        """
        self._theme = normalize_plotly_theme(theme)
        self._display_options.theme = self._theme
        self._sync_theme_to_plotly_dict()
        self._relayout_theme()

    def set_dark_mode(self, enabled: bool) -> None:
        """Set the Plotly layout theme from a dark-mode flag.

        Args:
            enabled: Whether dark mode is enabled.

        Returns:
            None.
        """
        self.set_theme("dark" if enabled else "light")

    def set_x_axis_limits(self, x_min: float | None, x_max: float | None) -> None:
        """Set x-axis limits programmatically.

        Args:
            x_min: Minimum x-axis value, or ``None`` for automatic scaling.
            x_max: Maximum x-axis value, or ``None`` for automatic scaling.

        Raises:
            ValueError: If both bounds are set and ``x_min >= x_max``.
        """
        new_range = (x_min, x_max)
        if _x_range_equal(new_range, (self._x_range.x_min, self._x_range.x_max)):
            self._last_applied_x_range = new_range
            return
        self._x_range = PlotlyAxisRange(x_min=x_min, x_max=x_max)
        self._last_applied_x_range = new_range
        xaxis = self._figure["layout"].setdefault("xaxis", {})
        if x_min is None or x_max is None:
            xaxis.pop("range", None)
            xaxis["autorange"] = True
            self._relayout({"xaxis.autorange": True})
            return
        xaxis["range"] = [float(x_min), float(x_max)]
        xaxis["autorange"] = False
        self._relayout({"xaxis.range": [float(x_min), float(x_max)], "xaxis.autorange": False})

    def reset_x_axis_limits(self) -> None:
        """Reset the x-axis to the full extent of visible line traces.

        When line traces are present, limits are derived from continuous traces
        only so scatter marker padding does not shift x=0. With no line traces,
        falls back to Plotly autorange.
        """
        derived = self._derive_x_range_from_visible_line_traces()
        if derived is not None:
            self._x_range = PlotlyAxisRange(x_min=None, x_max=None)
            self._last_applied_x_range = (None, None)
            self._push_x_axis_range_to_browser(*derived)
            return
        self.set_x_axis_limits(None, None)

    @property
    def x_range_limits(self) -> tuple[float | None, float | None]:
        """Return the widget's current logical x-axis limits."""
        return (self._x_range.x_min, self._x_range.x_max)

    def begin_select_x_range(self) -> None:
        """Enter one-shot box-select mode for user x-range selection.

        While armed, ``plotly_relayout`` payloads carrying ``selections`` x-bounds
        invoke ``on_x_range_selected`` once, then restore zoom mode.
        """
        self._x_range_selection_armed = True
        layout = self._figure.setdefault("layout", {})
        layout["dragmode"] = "select"
        self._relayout({"dragmode": "select"}, source="begin_select_x_range")

    def cancel_select_x_range(self) -> None:
        """Cancel box-select mode and restore zoom dragmode."""
        self._x_range_selection_armed = False
        layout = self._figure.setdefault("layout", {})
        layout["dragmode"] = "zoom"
        layout["selections"] = []
        self._relayout(
            {"dragmode": "zoom", "selections": []},
            source="cancel_select_x_range",
        )

    def add_measurement_line(
        self,
        *,
        name: str,
        orientation: str,
        value: float,
        visible: bool = True,
        y_axis: PlotlyYAxisSide = "left",
        editable: bool = True,
        color: str | None = None,
        dash: str = "dash",
        show_legend: bool = False,
        legend_label: str | None = None,
        on_changed: OnMeasurementChanged | None = None,
    ) -> MeasurementLine:
        """Add a horizontal or vertical measurement line.

        Args:
            name: Stable caller-defined measurement name.
            orientation: ``horizontal``/``h`` or ``vertical``/``v``.
            value: Initial line position in data coordinates.
            visible: Whether the line should be visible.
            y_axis: Y-axis for horizontal lines. ``"right"`` requires an
                existing ``layout.yaxis2`` from a right-axis trace or scatter.
            editable: Whether the user can drag the line.
            color: Plotly line color. ``None`` uses a theme-aware default.
            dash: Plotly dash style (``"solid"``, ``"dot"``, ``"dash"``, ...).
            show_legend: Whether the line appears in the Plotly legend.
            legend_label: Legend text when ``show_legend`` is True. Defaults to
                ``name``.
            on_changed: Optional per-measurement callback. Ignored when
                ``editable`` is False.

        Returns:
            Mutable measurement line object owned by the widget.

        Raises:
            ValueError: If the name already exists, orientation is invalid, or
                a right-axis horizontal line is requested before ``yaxis2`` exists.
        """
        clean = _validate_unique_name(
            name,
            self._measurements.get(str(name).strip()),
            label="measurement",
        )
        normalized = _normalize_orientation(orientation)
        axis = _normalize_y_axis_side(y_axis)
        if normalized == "vertical":
            axis = "left"
        elif axis == "right" and not self._has_yaxis2():
            raise ValueError(
                "cannot add right-axis measurement before a right-axis trace or scatter exists"
            )
        line_color = color if color is not None else self._default_measurement_color()
        line = MeasurementLine(
            name=clean,
            orientation=normalized,
            position=float(value),
            visible=bool(visible),
            y_axis=axis,
            editable=bool(editable),
            color=line_color,
            dash=str(dash),
            show_legend=bool(show_legend),
            legend_label=legend_label,
        )
        self._measurements[clean] = line
        if on_changed is not None and line.editable:
            self._measurement_callbacks[clean] = on_changed
        self._append_measurement_shape(
            clean,
            "line",
            1,
            normalized,
            float(value),
            visible,
            axis,
            editable=line.editable,
            color=line.color,
            dash=line.dash,
            show_legend=line.show_legend,
            legend_label=line.legend_label or clean,
        )
        self._push_shapes()
        return line

    def remove_measurement_line(self, name: str) -> None:
        """Remove a single-line measurement.

        Args:
            name: Existing single-line measurement name.

        Raises:
            KeyError: If the measurement does not exist.
            ValueError: If the measurement is a pair.
        """
        self._remove_measurement(name, expected_kind="line")

    def add_measurement_pair(
        self,
        *,
        name: str,
        orientation: str,
        value1: float,
        value2: float,
        visible: bool = True,
        y_axis: PlotlyYAxisSide = "left",
        on_changed: OnMeasurementChanged | None = None,
    ) -> MeasurementPair:
        """Add a draggable pair of horizontal or vertical measurement lines.

        Args:
            name: Stable caller-defined measurement-pair name.
            orientation: ``horizontal``/``h`` or ``vertical``/``v``.
            value1: Initial first-line position in data coordinates.
            value2: Initial second-line position in data coordinates.
            visible: Whether both lines should be visible.
            y_axis: Y-axis for horizontal lines. ``"right"`` requires an
                existing ``layout.yaxis2`` from a right-axis trace or scatter.
            on_changed: Optional per-measurement callback.

        Returns:
            Mutable measurement pair object owned by the widget.

        Raises:
            ValueError: If the name already exists, orientation is invalid, or
                a right-axis horizontal pair is requested before ``yaxis2`` exists.
        """
        clean = _validate_unique_name(
            name,
            self._measurements.get(str(name).strip()),
            label="measurement",
        )
        normalized = _normalize_orientation(orientation)
        axis = _normalize_y_axis_side(y_axis)
        if normalized == "vertical":
            axis = "left"
        elif axis == "right" and not self._has_yaxis2():
            raise ValueError(
                "cannot add right-axis measurement before a right-axis trace or scatter exists"
            )
        pair = MeasurementPair(
            name=clean,
            orientation=normalized,
            position1=float(value1),
            position2=float(value2),
            visible=bool(visible),
            y_axis=axis,
        )
        self._measurements[clean] = pair
        if on_changed is not None:
            self._measurement_callbacks[clean] = on_changed
        self._append_measurement_shape(
            clean, "pair", 1, normalized, float(value1), visible, axis
        )
        self._append_measurement_shape(
            clean, "pair", 2, normalized, float(value2), visible, axis
        )
        self._push_shapes()
        return pair

    def remove_measurement_pair(self, name: str) -> None:
        """Remove a paired-line measurement.

        Args:
            name: Existing measurement-pair name.

        Raises:
            KeyError: If the measurement does not exist.
            ValueError: If the measurement is a single line.
        """
        self._remove_measurement(name, expected_kind="pair")

    def _series_index(self, name: str, kind: _SeriesKind) -> int:
        """Return the current Plotly trace index for a named series."""
        for index, ref in enumerate(self._series_order):
            if ref.name == name and ref.kind == kind:
                return index
        raise KeyError(f"{kind} {name!r} does not exist")

    def _has_yaxis2(self) -> bool:
        """Return whether the figure layout currently defines ``yaxis2``."""
        layout = self._figure.get("layout", {})
        return isinstance(layout, dict) and "yaxis2" in layout

    def _has_right_axis_series(self) -> bool:
        """Return whether any trace or scatter is bound to the right y-axis."""
        return any(trace.y_axis == "right" for trace in self._traces.values()) or any(
            scatter.y_axis == "right" for scatter in self._scatters.values()
        )

    def _has_visible_right_axis_series(self) -> bool:
        """Return whether any visible trace or scatter uses the right y-axis."""
        return any(trace.y_axis == "right" and trace.visible for trace in self._traces.values()) or any(
            scatter.y_axis == "right" and scatter.visible for scatter in self._scatters.values()
        )

    def _yaxis2_decorations_visible(self) -> bool:
        """Return whether right y-axis title, ticks, and line should show."""
        return bool(self._display_options.show_y_axis_labels) and self._has_visible_right_axis_series()

    def _sync_yaxis2_from_series(self) -> None:
        """Create or remove ``layout.yaxis2`` based on right-axis traces/scatters."""
        if self._has_right_axis_series():
            self._ensure_yaxis2()
        else:
            self._maybe_remove_yaxis2()

    def _refresh_yaxis2_layout(self) -> None:
        """Update ``yaxis2`` decorations and right margin after visibility changes."""
        if self._has_right_axis_series():
            if not self._has_yaxis2():
                self._ensure_yaxis2()
                return
            layout = self._figure.setdefault("layout", {})
            layout["yaxis2"] = self._build_yaxis2_dict()
            self._sync_margins_to_plotly_dict()
            self._relayout_secondary_y_axis()
        else:
            self._maybe_remove_yaxis2()

    def _build_yaxis2_dict(self) -> dict[str, Any]:
        """Return a Plotly ``yaxis2`` layout dictionary for the current theme."""
        theme = theme_for_name(self._theme)
        visible = self._yaxis2_decorations_visible()
        yaxis2: dict[str, Any] = {
            "overlaying": "y",
            "side": "right",
            "autorange": True,
            "color": theme.axis_color,
            "linecolor": theme.axis_color,
            "tickcolor": theme.axis_color,
            "gridcolor": theme.grid_color,
            "zerolinecolor": theme.zero_line_color,
        }
        apply_axis_decorations(yaxis2, label_text=self._y2_label, visible=visible)
        return yaxis2

    def _ensure_yaxis2(self) -> None:
        """Ensure ``layout.yaxis2`` exists and matches current display options."""
        layout = self._figure.setdefault("layout", {})
        layout["yaxis2"] = self._build_yaxis2_dict()
        self._sync_margins_to_plotly_dict()
        self._relayout_secondary_y_axis()

    def _maybe_remove_yaxis2(self) -> None:
        """Remove ``layout.yaxis2`` when no right-axis traces or scatters remain."""
        if self._has_right_axis_series() or not self._has_yaxis2():
            return
        layout = self._figure.setdefault("layout", {})
        layout.pop("yaxis2", None)
        self._sync_margins_to_plotly_dict()
        self._relayout(
            {
                "yaxis2": None,
                "margin": dict(
                    layout.get(
                        "margin",
                        resolve_plot_layout_margins(
                            show_axis_labels=self._any_axis_labels_visible(),
                            show_legend=self._display_options.show_legend,
                            layout_margins_profile=self._layout_margins_profile,
                        ),
                    )
                ),
            },
            source="remove_yaxis2",
        )

    def _relayout_secondary_y_axis(self) -> None:
        """Push ``yaxis2`` and margin layout changes to the browser."""
        layout = self._figure.get("layout", {})
        yaxis2 = layout.get("yaxis2")
        if not isinstance(yaxis2, dict):
            return
        relayout: dict[str, Any] = {
            "yaxis2": yaxis2,
            "margin": dict(
                layout.get(
                    "margin",
                    resolve_plot_layout_margins(
                        show_axis_labels=self._any_axis_labels_visible(),
                        show_legend=self._display_options.show_legend,
                        has_yaxis2=self._yaxis2_decorations_visible(),
                        layout_margins_profile=self._layout_margins_profile,
                    ),
                )
            ),
        }
        self._relayout(relayout, source="yaxis2")

    def _remove_measurement(self, name: str, *, expected_kind: _MeasurementKind) -> None:
        """Remove a measurement and all associated Plotly shapes."""
        clean = str(name).strip()
        measurement = self._measurements.get(clean)
        if measurement is None:
            raise KeyError(f"measurement {clean!r} does not exist")
        is_pair = isinstance(measurement, MeasurementPair)
        if expected_kind == "pair" and not is_pair:
            raise ValueError(f"measurement {clean!r} is not a pair")
        if expected_kind == "line" and is_pair:
            raise ValueError(f"measurement {clean!r} is not a single line")
        self._measurements.pop(clean)
        self._measurement_callbacks.pop(clean, None)
        keep_shapes: list[dict[str, Any]] = []
        keep_refs: list[_ShapeRef] = []
        for shape, ref in zip(self._shapes(), self._shape_refs, strict=True):
            if ref.name == clean:
                continue
            keep_shapes.append(shape)
            keep_refs.append(ref)
        self._figure["layout"]["shapes"] = keep_shapes
        self._shape_refs = keep_refs
        self._push_shapes()

    def _trace_to_plotly(self, data: PlotlyTraceData) -> dict[str, Any]:
        """Return a Plotly ``scattergl`` line trace dictionary."""
        hoverinfo = "all" if self._display_options.show_hover_info else "skip"
        trace: dict[str, Any] = {
            "type": "scattergl",
            "mode": "lines",
            "name": data.name,
            "x": list(data.x),
            "y": list(data.y),
            "visible": True if data.visible else False,
            "hoverinfo": hoverinfo,
        }
        if data.line_color is not None or data.line_dash is not None:
            line: dict[str, str] = {}
            if data.line_color is not None:
                line["color"] = data.line_color
            if data.line_dash is not None:
                line["dash"] = data.line_dash
            trace["line"] = line
        if data.y_axis == "right":
            trace["yaxis"] = "y2"
        return trace

    def _scatter_to_plotly(self, data: PlotlyScatterData) -> dict[str, Any]:
        """Return a Plotly ``scattergl`` marker trace dictionary."""
        hoverinfo = "all" if self._display_options.show_hover_info else "skip"
        trace: dict[str, Any] = {
            "type": "scattergl",
            "mode": "markers",
            "name": data.name,
            "x": list(data.x),
            "y": list(data.y),
            "visible": True if data.visible else False,
            "hoverinfo": hoverinfo,
            "cliponaxis": True,
            "marker": {"size": 8},
        }
        if data.y_axis == "right":
            trace["yaxis"] = "y2"
        return trace

    def _append_measurement_shape(
        self,
        name: str,
        kind: _MeasurementKind,
        line_number: int,
        orientation: PlotlyLineOrientation,
        value: float,
        visible: bool,
        y_axis: PlotlyYAxisSide = "left",
        *,
        editable: bool = True,
        color: str | None = None,
        dash: str = "dash",
        show_legend: bool = False,
        legend_label: str | None = None,
    ) -> None:
        """Append one Plotly layout shape for a measurement line."""
        shape = self._line_shape(
            orientation=orientation,
            value=value,
            visible=visible,
            y_axis=y_axis,
            editable=editable,
            color=color if color is not None else self._default_measurement_color(),
            dash=dash,
            show_legend=show_legend,
            legend_label=legend_label or name,
        )
        self._shapes().append(shape)
        self._shape_refs.append(_ShapeRef(name=name, kind=kind, line_number=line_number))

    def _default_measurement_color(self) -> str:
        """Return a high-contrast measurement line color for the current theme.

        Returns:
            Plotly color string.
        """
        return theme_for_name(self._theme).font_color

    def _line_shape(
        self,
        *,
        orientation: PlotlyLineOrientation,
        value: float,
        visible: bool,
        y_axis: PlotlyYAxisSide = "left",
        editable: bool = True,
        color: str,
        dash: str = "dash",
        show_legend: bool = False,
        legend_label: str,
    ) -> dict[str, Any]:
        """Build one Plotly line shape."""
        line_style = {"width": 3, "dash": str(dash), "color": str(color)}
        if orientation == "horizontal":
            shape: dict[str, Any] = {
                "type": "line",
                "xref": "paper",
                "x0": 0,
                "x1": 1,
                "yref": "y2" if y_axis == "right" else "y",
                "y0": value,
                "y1": value,
                "visible": bool(visible),
                "editable": bool(editable),
                "line": line_style,
            }
        else:
            shape = {
                "type": "line",
                "xref": "x",
                "x0": value,
                "x1": value,
                "yref": "paper",
                "y0": 0,
                "y1": 1,
                "visible": bool(visible),
                "editable": bool(editable),
                "line": line_style,
            }
        # Omit legend keys unless requested. Shape legend/name can change Plotly
        # edit interaction away from whole-shape drag under shapePosition.
        if show_legend:
            shape["name"] = str(legend_label)
            shape["showlegend"] = True
        return shape

    def _shapes(self) -> list[dict[str, Any]]:
        """Return the mutable layout shape list."""
        layout = self._figure.setdefault("layout", {})
        shapes = layout.setdefault("shapes", [])
        if not isinstance(shapes, list):
            raise TypeError("Plotly layout.shapes must be a list")
        return shapes

    def _register_self_relayout(
        self,
        payload: dict[str, object],
        *,
        source: str,
        echo_suppressions: int = 1,
    ) -> None:
        """Register a self-initiated relayout payload for short-lived echo suppression.

        Args:
            payload: Relayout key/value pairs expected back from Plotly.
            source: Short label for debugging.
            echo_suppressions: Matching relayout callbacks to suppress before drop.

        Returns:
            None.
        """
        now = time.perf_counter()
        expires_at = now + _SELF_RELAYOUT_TTL_SEC
        self._pending_self_relayouts = [
            item
            for item in self._pending_self_relayouts
            if float(item["expires_at"]) >= now
        ]
        self._pending_self_relayouts.append(
            {
                "source": source,
                "expected": dict(payload),
                "expires_at": expires_at,
                "remaining": max(1, int(echo_suppressions)),
            }
        )

    def _pop_matching_self_relayout(self, args: dict[str, object]) -> str | None:
        """Match incoming relayout to a pending self-initiated payload.

        Args:
            args: Incoming Plotly relayout payload.

        Returns:
            Source label when matched, otherwise ``None``.
        """
        now = time.perf_counter()
        self._pending_self_relayouts = [
            item
            for item in self._pending_self_relayouts
            if float(item["expires_at"]) >= now
        ]
        for idx, item in enumerate(self._pending_self_relayouts):
            expected = item["expected"]
            if not isinstance(expected, dict):
                continue
            if all(args.get(key) == value for key, value in expected.items()):
                source = str(item["source"])
                remaining = int(item.get("remaining", 1)) - 1
                if remaining <= 0:
                    self._pending_self_relayouts.pop(idx)
                else:
                    self._pending_self_relayouts[idx]["remaining"] = remaining
                return source
        return None

    def _handle_x_range_selection_relayout(self, args: dict[str, Any]) -> bool:
        """Consume a box-select relayout while selection mode is armed.

        Args:
            args: Plotly relayout event payload.

        Returns:
            ``True`` when the payload was handled as a completed selection.
        """
        if not self._x_range_selection_armed:
            return False
        x0, x1 = extract_rect_selection_x_range_from_relayout(args)
        if x0 is None or x1 is None:
            return False
        self._x_range_selection_armed = False
        layout = self._figure.setdefault("layout", {})
        layout["dragmode"] = "zoom"
        layout["selections"] = []
        self._relayout(
            {"dragmode": "zoom", "selections": []},
            source="selection_complete",
        )
        if self._on_x_range_selected is not None:
            self._on_x_range_selected(float(x0), float(x1))
        return True

    def _should_emit_user_x_range_relayout(self, args: dict[str, Any]) -> bool:
        """Return whether ``args`` looks like a user x-axis range gesture."""
        if self._parse_x_range_event(args) is None:
            return False
        if args.get("xaxis.autorange") is True:
            return True
        if not _relayout_has_axis_range(args):
            return False
        if _is_normalized_only_relayout(args):
            return False
        return _relayout_has_bracket_axis_range(args)

    def _on_plotly_relayout(self, event: Any) -> None:
        """Handle Plotly relayout events from user zooms, selections, and shape drags."""
        args = getattr(event, "args", None)
        if not isinstance(args, dict):
            return

        # logger.info("plotly_relayout args=%s", args)

        if self._ignore_relayout:
            return
        if self._pop_matching_self_relayout(args) is not None:
            return
        if self._handle_x_range_selection_relayout(args):
            return
        self._sync_shape_edits(args)
        if self._should_emit_user_x_range_relayout(args):
            self._emit_x_range_if_needed(args)

    def _on_plotly_doubleclick(self, event: Any) -> None:
        """Reset x-axis limits after Plotly double-click autorange.

        Args:
            event: NiceGUI double-click event (unused).
        """
        _ = event
        self.reset_x_axis_limits()
        if self._on_x_range_changed is not None:
            self._on_x_range_changed(None, None)

    def _emit_x_range_if_needed(self, args: dict[str, Any]) -> None:
        """Emit x-range callback for user axis range changes."""
        parsed = self._parse_x_range_event(args)
        if parsed is None:
            return
        if self._is_x_range_echo(parsed):
            return
        xaxis = self._figure["layout"].setdefault("xaxis", {})
        x_min, x_max = parsed
        if x_min is None or x_max is None:
            xaxis.pop("range", None)
            xaxis["autorange"] = True
        else:
            xaxis["range"] = [x_min, x_max]
            xaxis["autorange"] = False
        if self._on_x_range_changed is not None:
            self._on_x_range_changed(x_min, x_max)
        self._last_applied_x_range = parsed

    def _is_x_range_echo(
        self, new_range: tuple[float | None, float | None]
    ) -> bool:
        """Return whether ``new_range`` echoes the last programmatic apply.

        Args:
            new_range: Candidate ``(x_min, x_max)`` from a relayout event.

        Returns:
            ``True`` when both values match the last applied pair within
            tolerance.
        """
        last = self._last_applied_x_range
        if last is None:
            return False
        return _x_range_equal(last, new_range)

    @staticmethod
    def _parse_x_range_event(args: dict[str, Any]) -> tuple[float | None, float | None] | None:
        """Parse a Plotly relayout payload for x-axis range changes.

        Args:
            args: Plotly relayout event payload.

        Returns:
            ``(x_min, x_max)``, ``(None, None)`` for autorange, or ``None``
            when the event does not describe an x-axis range change.
        """
        if args.get("xaxis.autorange") is True:
            return (None, None)
        if "xaxis.range" in args:
            value = args["xaxis.range"]
            if isinstance(value, Sequence) and len(value) == 2:
                return (float(value[0]), float(value[1]))
        if "xaxis.range[0]" in args and "xaxis.range[1]" in args:
            return (float(args["xaxis.range[0]"]), float(args["xaxis.range[1]"]))
        return None

    def _sync_shape_edits(self, args: dict[str, Any]) -> None:
        """Mirror user-dragged shape coordinates and emit measurement callbacks."""
        changed_indices = self._shape_indices_from_relayout(args)
        if not changed_indices:
            return
        shapes = self._shapes()
        needs_shape_push = False
        for index in changed_indices:
            if index >= len(shapes) or index >= len(self._shape_refs):
                continue
            shape = shapes[index]
            self._apply_shape_args(shape, index, args)
            ref = self._shape_refs[index]
            measurement = self._measurements.get(ref.name)
            if measurement is None:
                continue
            if isinstance(measurement, MeasurementLine) and not measurement.editable:
                continue
            position = self._measurement_position_after_edit(
                shape,
                measurement.orientation,
                index=index,
                args=args,
            )
            if isinstance(measurement, MeasurementLine):
                measurement.position = position
                self._normalize_measurement_shape(shape, measurement)
                needs_shape_push = True
                event = MeasurementChangeEvent(
                    name=measurement.name,
                    kind="line",
                    orientation=measurement.orientation,
                    position=position,
                    y_axis=measurement.y_axis,
                )
            else:
                if ref.line_number == 1:
                    measurement.position1 = position
                else:
                    measurement.position2 = position
                event = MeasurementChangeEvent(
                    name=measurement.name,
                    kind="pair",
                    orientation=measurement.orientation,
                    position=position,
                    position1=measurement.position1,
                    position2=measurement.position2,
                    delta=measurement.delta,
                    y_axis=measurement.y_axis,
                )
            self._emit_measurement_changed(event)
        if needs_shape_push:
            self._push_shapes()

    @staticmethod
    def _shape_indices_from_relayout(args: dict[str, Any]) -> set[int]:
        """Return shape indices touched by a relayout payload."""
        indices: set[int] = set()
        for key in args:
            if key.startswith("shapes["):
                close = key.find("]")
                if close > len("shapes["):
                    try:
                        indices.add(int(key[len("shapes[") : close]))
                    except ValueError:
                        continue
        if not indices and isinstance(args.get("shapes"), list):
            indices.update(range(len(args["shapes"])))
        return indices

    @staticmethod
    def _apply_shape_args(shape: dict[str, Any], index: int, args: dict[str, Any]) -> None:
        """Apply relayout payload shape keys to one local shape dictionary."""
        full_shapes = args.get("shapes")
        if isinstance(full_shapes, list) and index < len(full_shapes) and isinstance(full_shapes[index], dict):
            shape.clear()
            shape.update(full_shapes[index])
            return
        prefix = f"shapes[{index}]."
        for key, value in args.items():
            if key.startswith(prefix):
                shape[key[len(prefix) :]] = value

    @staticmethod
    def _shape_position(shape: dict[str, Any], orientation: PlotlyLineOrientation) -> float:
        """Return the data-coordinate position for a Plotly line shape."""
        if orientation == "horizontal":
            y0 = float(shape.get("y0", shape.get("y1", 0.0)))
            y1 = float(shape.get("y1", y0))
            return (y0 + y1) / 2.0
        x0 = float(shape.get("x0", shape.get("x1", 0.0)))
        x1 = float(shape.get("x1", x0))
        return (x0 + x1) / 2.0

    @classmethod
    def _measurement_position_after_edit(
        cls,
        shape: dict[str, Any],
        orientation: PlotlyLineOrientation,
        *,
        index: int,
        args: dict[str, Any],
    ) -> float:
        """Return the post-drag position for a measurement line.

        Plotly line shapes expose two endpoints. Vertex drags often update only
        ``y0`` or only ``y1`` (or ``x0`` / ``x1``). Prefer the endpoint value(s)
        present in the relayout payload so dragging either handle moves the
        line; then callers normalize back to a single axis-aligned line.

        Args:
            shape: Shape dict after :meth:`_apply_shape_args`.
            orientation: Line orientation.
            index: Shape index in ``layout.shapes``.
            args: Raw Plotly relayout payload.

        Returns:
            Data-coordinate position for the measurement.
        """
        prefix = f"shapes[{index}]."
        if orientation == "horizontal":
            keys = (f"{prefix}y0", f"{prefix}y1")
        else:
            keys = (f"{prefix}x0", f"{prefix}x1")
        changed = [float(args[key]) for key in keys if key in args]
        if len(changed) == 1:
            return changed[0]
        if len(changed) == 2:
            return (changed[0] + changed[1]) / 2.0
        return cls._shape_position(shape, orientation)

    @staticmethod
    def _normalize_measurement_shape(
        shape: dict[str, Any],
        measurement: MeasurementLine,
    ) -> None:
        """Keep single-line measurements axis-aligned after a drag.

        Args:
            shape: Mutable Plotly shape dict that was edited.
            measurement: Owning measurement line.

        Returns:
            None.
        """
        position = float(measurement.position)
        if measurement.orientation == "horizontal":
            shape["y0"] = position
            shape["y1"] = position
            # Keep full-width paper span so the line stays a true H-line.
            shape["xref"] = "paper"
            shape["x0"] = 0
            shape["x1"] = 1
            return
        shape["x0"] = position
        shape["x1"] = position
        shape["yref"] = "paper"
        shape["y0"] = 0
        shape["y1"] = 1

    def _emit_measurement_changed(self, event: MeasurementChangeEvent) -> None:
        """Invoke global and per-measurement callbacks for a measurement change."""
        callback = self._measurement_callbacks.get(event.name)
        if callback is not None:
            callback(event)
        if self._on_measurement_changed is not None:
            self._on_measurement_changed(event)

    def _js_plotly_graph_div(self) -> str:
        """Return JavaScript that resolves this NiceGUI Plotly graph div."""
        plot_id = self._plot_element.id
        return f"""const host = getElement({plot_id}).$el;
if (!host) return;
const plotDiv = host.querySelector('.js-plotly-plot') || host;
if (!plotDiv || !plotDiv.data) return;
"""

    def _add_plotly_trace(self, trace: dict[str, Any]) -> None:
        """Push a newly added trace to the browser."""
        js = f"""
{self._js_plotly_graph_div()}
Plotly.addTraces(plotDiv, [{json.dumps(trace)}]);
"""
        self._run_plotly_javascript(js)

    def _restyle_plotly_trace(self, index: int, trace: dict[str, Any]) -> None:
        """Push trace replacement values to the browser with ``Plotly.restyle``."""
        restyle = {key: [value] for key, value in trace.items() if key != "type"}
        js = f"""
{self._js_plotly_graph_div()}
Plotly.restyle(plotDiv, {json.dumps(restyle)}, [{index}]);
"""
        self._run_plotly_javascript(js)

    def _delete_plotly_trace(self, index: int) -> None:
        """Remove one Plotly trace from the browser."""
        js = f"""
{self._js_plotly_graph_div()}
Plotly.deleteTraces(plotDiv, [{index}]);
"""
        self._run_plotly_javascript(js)

    def _relayout(self, payload: dict[str, Any], *, source: str = "relayout") -> None:
        """Push a Plotly relayout payload to the browser."""
        self._register_self_relayout(payload, source=source)
        js = f"""
{self._js_plotly_graph_div()}
Plotly.relayout(plotDiv, {json.dumps(payload)});
"""
        self._run_plotly_javascript(js)

    def _push_shapes(self) -> None:
        """Push the current layout shapes to the browser."""
        self._apply_event_overlays()

    def _apply_event_overlays(self) -> None:
        """Merge measurement shapes with event overlays and relayout."""
        measurement_shapes = self._shapes()[: len(self._shape_refs)]
        combined = measurement_shapes + self.events.build_plotly_shapes()
        self._figure["layout"]["shapes"] = combined
        self._relayout({"shapes": combined}, source="event_overlays")

    def _sync_theme_to_plotly_dict(self) -> None:
        """Synchronize the selected light/dark theme into the local figure dict."""
        layout = self._figure.setdefault("layout", {})
        if not isinstance(layout, dict):
            layout = {}
            self._figure["layout"] = layout
        apply_plotly_theme_to_layout(layout, self._theme)

    def _sync_axis_labels_to_plotly_dict(self) -> None:
        """Synchronize axis decoration visibility into the local figure dict."""
        layout = self._figure.setdefault("layout", {})
        axis_specs = (
            ("xaxis", self._x_label, self._display_options.show_x_axis_labels),
            ("yaxis", self._y_label, self._display_options.show_y_axis_labels),
        )
        for axis_name, label_text, visible in axis_specs:
            axis = layout.setdefault(axis_name, {})
            if not isinstance(axis, dict):
                axis = {}
                layout[axis_name] = axis
            apply_axis_decorations(axis, label_text=label_text, visible=bool(visible))
        if self._has_yaxis2():
            yaxis2 = layout.setdefault("yaxis2", self._build_yaxis2_dict())
            if isinstance(yaxis2, dict):
                deco_visible = self._yaxis2_decorations_visible()
                apply_axis_decorations(
                    yaxis2,
                    label_text=self._y2_label,
                    visible=deco_visible,
                )
                yaxis2["showgrid"] = False

    def _sync_margins_to_plotly_dict(self) -> None:
        """Synchronize layout margins with axis-label and legend visibility."""
        layout = self._figure.setdefault("layout", {})
        layout["margin"] = resolve_plot_layout_margins(
            show_axis_labels=self._any_axis_labels_visible(),
            show_legend=bool(self._display_options.show_legend),
            has_yaxis2=self._yaxis2_decorations_visible(),
            layout_margins_profile=self._layout_margins_profile,
        )

    def _sync_axis_stabilization_to_plotly_dict(self) -> None:
        """Apply stack-profile axis stabilization into the local figure dict."""
        if self._layout_margins_profile is None:
            return
        layout = self._figure.setdefault("layout", {})
        if isinstance(layout, dict):
            self._layout_margins_profile.apply_axis_stabilization(layout)

    def _sync_legend_to_plotly_dict(self) -> None:
        """Synchronize legend visibility and bottom horizontal layout into the figure dict."""
        layout = self._figure.setdefault("layout", {})
        layout["showlegend"] = bool(self._display_options.show_legend)
        legend = layout.setdefault("legend", {})
        if not isinstance(legend, dict):
            legend = {}
            layout["legend"] = legend
        if self._display_options.show_legend:
            legend.update(dict(_PLOTLY_PLOT_LEGEND))

    def _sync_plotly_config_to_plotly_dict(self) -> None:
        """Synchronize Plotly config options into the local figure dict."""
        config = self._figure.setdefault("config", {})
        if not isinstance(config, dict):
            config = {}
            self._figure["config"] = config
        config["displayModeBar"] = bool(self._display_options.show_plotly_toolbar)
        config["editable"] = True
        config["edits"] = {
            "shapePosition": True,
            "titleText": False,
            "axisTitleText": False,
            "legendText": False,
            "legendPosition": False,
        }

    def _sync_hover_info_to_plotly_dict(self) -> None:
        """Synchronize hover-info visibility into all trace dictionaries."""
        hoverinfo = "all" if self._display_options.show_hover_info else "skip"
        for trace in self._figure.get("data", []):
            if isinstance(trace, dict):
                trace["hoverinfo"] = hoverinfo

    def _restyle_hover_info(self) -> None:
        """Push hover-info changes to the browser via ``Plotly.restyle``."""
        if not self._figure.get("data"):
            return
        hoverinfo = "all" if self._display_options.show_hover_info else "skip"
        indices = list(range(len(self._figure["data"])))
        js = f"""
{self._js_plotly_graph_div()}
Plotly.restyle(plotDiv, {{hoverinfo: {json.dumps(hoverinfo)}}}, {json.dumps(indices)});
"""
        self._run_plotly_javascript(js)

    def _react_plotly_config(self) -> None:
        """Push Plotly config changes to the browser."""
        config = self._figure.get("config", {})
        js = f"""
{self._js_plotly_graph_div()}
Plotly.react(plotDiv, plotDiv.data, plotDiv.layout, {json.dumps(config)});
"""
        self._run_plotly_javascript(js)

    def _relayout_axis_labels_and_margins(self) -> None:
        """Push axis-label and margin layout changes to the browser."""
        layout = self._figure.get("layout", {})
        relayout: dict[str, Any] = {"margin": layout.get("margin", {})}
        for axis_name in ("xaxis", "yaxis"):
            axis = layout.get(axis_name, {})
            if not isinstance(axis, dict):
                continue
            title = axis.get("title", {})
            if isinstance(title, dict):
                relayout[f"{axis_name}.title.text"] = title.get("text", "")
                relayout[f"{axis_name}.title.font.size"] = PLOTLY_AXIS_LABEL_FONT_SIZE
            relayout[f"{axis_name}.tickfont.size"] = PLOTLY_AXIS_LABEL_FONT_SIZE
            relayout[f"{axis_name}.showticklabels"] = axis.get("showticklabels", False)
            relayout[f"{axis_name}.ticks"] = axis.get("ticks", "")
            relayout[f"{axis_name}.showline"] = axis.get("showline", False)
            relayout[f"{axis_name}.zeroline"] = axis.get("zeroline", False)
            relayout[f"{axis_name}.showgrid"] = axis.get("showgrid", False)
            if "automargin" in axis:
                relayout[f"{axis_name}.automargin"] = axis.get("automargin")
        yaxis2 = layout.get("yaxis2")
        if isinstance(yaxis2, dict):
            title = yaxis2.get("title", {})
            if isinstance(title, dict):
                relayout["yaxis2.title.text"] = title.get("text", "")
                relayout["yaxis2.title.font.size"] = PLOTLY_AXIS_LABEL_FONT_SIZE
            relayout["yaxis2.tickfont.size"] = PLOTLY_AXIS_LABEL_FONT_SIZE
            relayout["yaxis2.showticklabels"] = yaxis2.get("showticklabels", False)
            relayout["yaxis2.ticks"] = yaxis2.get("ticks", "")
            relayout["yaxis2.showline"] = yaxis2.get("showline", False)
            relayout["yaxis2.showgrid"] = yaxis2.get("showgrid", False)
        self._relayout(relayout)

    def _relayout_legend(self) -> None:
        """Push legend visibility, layout, and bottom margin to the browser."""
        layout = self._figure.get("layout", {})
        relayout: dict[str, Any] = {
            "showlegend": bool(layout.get("showlegend", True)),
            "margin": layout.get("margin", {}),
        }
        if relayout["showlegend"]:
            legend = layout.get("legend")
            if isinstance(legend, dict):
                relayout["legend"] = legend
        self._relayout(relayout, source="legend_visible")

    def _relayout_theme(self) -> None:
        """Push light/dark theme layout properties to the browser."""
        layout = self._figure.setdefault("layout", {})
        if not isinstance(layout, dict):
            return
        theme = theme_for_name(self._theme)
        relayout: dict[str, Any] = {
            "paper_bgcolor": theme.paper_bgcolor,
            "plot_bgcolor": theme.plot_bgcolor,
            "font.color": theme.font_color,
        }
        for axis_name in ("xaxis", "yaxis"):
            axis = layout.get(axis_name, {})
            if not isinstance(axis, dict):
                continue
            relayout[f"{axis_name}.color"] = axis.get("color", theme.axis_color)
            relayout[f"{axis_name}.linecolor"] = axis.get("linecolor", theme.axis_color)
            relayout[f"{axis_name}.tickcolor"] = axis.get("tickcolor", theme.axis_color)
            relayout[f"{axis_name}.gridcolor"] = axis.get("gridcolor", theme.grid_color)
            relayout[f"{axis_name}.zerolinecolor"] = axis.get("zerolinecolor", theme.zero_line_color)
        yaxis2 = layout.get("yaxis2")
        if isinstance(yaxis2, dict):
            relayout["yaxis2.color"] = yaxis2.get("color", theme.axis_color)
            relayout["yaxis2.linecolor"] = yaxis2.get("linecolor", theme.axis_color)
            relayout["yaxis2.tickcolor"] = yaxis2.get("tickcolor", theme.axis_color)
            relayout["yaxis2.gridcolor"] = yaxis2.get("gridcolor", theme.grid_color)
            relayout["yaxis2.zerolinecolor"] = yaxis2.get("zerolinecolor", theme.zero_line_color)
        self._relayout(relayout)

    def _push_series_data(self) -> None:
        """Push the full trace/scatter data array to the browser in one update."""
        data_json = json.dumps(self._figure["data"])
        js = f"""
{self._js_plotly_graph_div()}
const newData = {data_json};
const oldCount = plotDiv.data ? plotDiv.data.length : 0;
if (oldCount > 0) {{
  Plotly.deleteTraces(plotDiv, [...Array(oldCount).keys()]);
}}
if (newData.length > 0) {{
  Plotly.addTraces(plotDiv, newData);
}}
"""
        self._run_plotly_javascript(js)
        self._plot_element.update()

    def _run_plotly_javascript(self, js: str) -> None:
        """Run Plotly JavaScript while suppressing programmatic relayout echo.

        NiceGUI cannot schedule browser JavaScript until its event loop exists.
        Demo scripts commonly populate widgets before ``ui.run()`` starts that
        loop, so the local figure dictionary remains the source of truth and the
        browser receives the complete state during initial rendering. Incremental
        JavaScript pushes are only needed after the client is live.

        Args:
            js: JavaScript source to execute in the owning browser client.
        """
        if core.loop is None and self._plot_element.client.__class__.__module__.startswith("nicegui"):
            logger.debug("Skipping Plotly JavaScript update before NiceGUI loop starts.")
            return

        self._ignore_relayout = True
        try:
            self._plot_element.client.run_javascript(js, timeout=2.0)
        except RuntimeError:
            logger.warning("Could not run Plotly JavaScript; browser client unavailable.")
        except AssertionError:
            logger.debug("Skipping Plotly JavaScript update before NiceGUI loop starts.")
        except Exception:
            logger.exception("Failed to run Plotly JavaScript update.")
        finally:
            self._ignore_relayout = False
