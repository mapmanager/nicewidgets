"""Shared Plotly axis decoration and layout-margin helpers for nicewidgets."""

from __future__ import annotations

from typing import Any

from nicewidgets.plotly_layout_margins import PlotlyLayoutMarginsProfile

PLOTLY_AXIS_LABEL_FONT_SIZE: int = 11

_PLOTLY_MARGIN_EDGE_COMPACT: int = 8
_PLOTLY_MARGIN_L_WITH_AXIS_LABELS: int = 60
_PLOTLY_MARGIN_R_WITH_AXIS_LABELS: int = 24
_PLOTLY_MARGIN_R_WITH_DUAL_Y_AXIS_LABELS: int = 60
_PLOTLY_MARGIN_T_WITH_AXIS_LABELS: int = 10
_PLOTLY_MARGIN_B_WITH_AXIS_LABELS: int = 40
_PLOTLY_MARGIN_B_WITH_LEGEND: int = 40
_PLOTLY_MARGIN_B_WITH_AXIS_LABELS_AND_LEGEND: int = 72


def any_axis_labels_visible(*, show_x_axis_labels: bool, show_y_axis_labels: bool) -> bool:
    """Return whether any primary axis decorations are visible.

    Args:
        show_x_axis_labels: Whether x-axis decorations are visible.
        show_y_axis_labels: Whether y-axis decorations are visible.

    Returns:
        True when at least one axis has visible decorations.
    """
    return bool(show_x_axis_labels or show_y_axis_labels)


def apply_axis_label_font(axis: dict[str, Any]) -> None:
    """Set explicit axis title and tick label font sizes on ``axis``.

    Args:
        axis: Plotly ``xaxis`` / ``yaxis`` layout dictionary to update in place.

    Returns:
        None.
    """
    title = axis.setdefault("title", {})
    if not isinstance(title, dict):
        title = {}
        axis["title"] = title
    font = title.setdefault("font", {})
    if not isinstance(font, dict):
        font = {}
        title["font"] = font
    font["size"] = PLOTLY_AXIS_LABEL_FONT_SIZE
    tickfont = axis.setdefault("tickfont", {})
    if not isinstance(tickfont, dict):
        tickfont = {}
        axis["tickfont"] = tickfont
    tickfont["size"] = PLOTLY_AXIS_LABEL_FONT_SIZE


def apply_axis_decorations(
    axis: dict[str, Any],
    *,
    label_text: str,
    visible: bool,
) -> None:
    """Apply axis title, tick, and line visibility plus label font size.

    Grid lines remain off to match :class:`PlotlyPlotWidget` defaults.

    Args:
        axis: Plotly axis layout dictionary to update in place.
        label_text: Axis title text when ``visible`` is true.
        visible: Whether axis decorations should be visible.

    Returns:
        None.
    """
    apply_axis_label_font(axis)
    title = axis.setdefault("title", {})
    if not isinstance(title, dict):
        title = {}
        axis["title"] = title
    title["text"] = label_text if visible else ""
    axis["showticklabels"] = visible
    axis["ticks"] = "outside" if visible else ""
    axis["showline"] = visible
    axis["zeroline"] = False
    axis["showgrid"] = False


def resolve_plot_layout_margins(
    *,
    show_axis_labels: bool,
    show_legend: bool,
    has_yaxis2: bool = False,
    layout_margins_profile: PlotlyLayoutMarginsProfile | None = None,
) -> dict[str, int]:
    """Return Plotly layout margins for axis-label and legend visibility.

    Args:
        show_axis_labels: Whether any axis decorations are visible.
        show_legend: Whether the bottom horizontal legend is visible.
        has_yaxis2: Whether a secondary right y-axis is present.
        layout_margins_profile: Optional fixed margin profile that bypasses
            widget-default margin tables.

    Returns:
        Plotly ``layout.margin`` dictionary with ``l``, ``r``, ``t``, and ``b``.
    """
    if layout_margins_profile is not None:
        return layout_margins_profile.resolve(show_axis_labels=show_axis_labels)
    if show_axis_labels:
        left = _PLOTLY_MARGIN_L_WITH_AXIS_LABELS
        right = (
            _PLOTLY_MARGIN_R_WITH_DUAL_Y_AXIS_LABELS
            if has_yaxis2
            else _PLOTLY_MARGIN_R_WITH_AXIS_LABELS
        )
        top = _PLOTLY_MARGIN_T_WITH_AXIS_LABELS
    else:
        left = right = top = _PLOTLY_MARGIN_EDGE_COMPACT

    if show_axis_labels and show_legend:
        bottom = _PLOTLY_MARGIN_B_WITH_AXIS_LABELS_AND_LEGEND
    elif show_axis_labels:
        bottom = _PLOTLY_MARGIN_B_WITH_AXIS_LABELS
    elif show_legend:
        bottom = _PLOTLY_MARGIN_B_WITH_LEGEND
    else:
        bottom = _PLOTLY_MARGIN_EDGE_COMPACT

    return {"l": left, "r": right, "t": top, "b": bottom}
