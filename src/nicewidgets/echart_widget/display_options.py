"""Display options for the ECharts widget."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EChartDisplayOptions:
    """User-facing display toggles for :class:`EChartWidget`.

    Args:
        show_toolbar: Whether ECharts' toolbox (zoom/restore/brush icons) is
            visible above the chart. Defaults to ``True`` so the chart's
            zoom/restore/brush actions are discoverable; users can hide it via
            the right-click context menu.
        show_hover_info: Whether the ECharts ``tooltip`` floating layer (hover
            label with x/y values) is shown. Maps to ECharts' ``tooltip.show``
            option. Defaults to ``False`` so the chart starts uncluttered;
            users can toggle it via the right-click context menu.
        show_axis_labels: Whether axis decorations (axis name, tick labels,
            tick marks, and axis line) are shown on both axes. Maps to
            ECharts' ``axisLabel.show`` / ``axisTick.show`` / ``axisLine.show``
            (and blanks the axis ``name`` when off). Defaults to ``True``.
        show_horizontal_lines: Whether horizontal grid lines are drawn at the
            y-axis tick positions. Maps to ECharts' ``yAxis.splitLine.show``.
            Defaults to ``False``.
        show_vertical_lines: Whether vertical grid lines are drawn at the
            x-axis tick positions. Maps to ECharts' ``xAxis.splitLine.show``.
            Defaults to ``False``.
    """

    show_toolbar: bool = True
    show_hover_info: bool = False
    show_axis_labels: bool = True
    show_horizontal_lines: bool = False
    show_vertical_lines: bool = False
