"""Display options for the reusable Plotly plot widget."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from nicewidgets.plotly_theme import PlotlyThemeName, normalize_plotly_theme


@dataclass(slots=True)
class PlotlyPlotDisplayOptions:
    """User-facing display toggles for :class:`PlotlyPlotWidget`.

    Args:
        show_x_axis_labels: Whether x-axis title text, tick labels, ticks, axis
            lines, and grid lines are visible.
        show_y_axis_labels: Whether primary left and secondary right y-axis title
            text, tick labels, ticks, axis lines, and grid lines are visible.
        show_plotly_toolbar: Whether Plotly's modebar is visible.
        show_hover_info: Whether Plotly emits hover labels for plot traces.
        show_legend: Whether the Plotly legend is visible.
        theme: Plotly layout color theme.
    """

    show_x_axis_labels: bool = False
    show_y_axis_labels: bool = False
    show_plotly_toolbar: bool = False
    show_hover_info: bool = False
    show_legend: bool = True
    theme: PlotlyThemeName = "light"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of these options.

        Returns:
            Mapping with one entry per field. ``theme`` is a plain string.
        """
        return {
            "show_x_axis_labels": bool(self.show_x_axis_labels),
            "show_y_axis_labels": bool(self.show_y_axis_labels),
            "show_plotly_toolbar": bool(self.show_plotly_toolbar),
            "show_hover_info": bool(self.show_hover_info),
            "show_legend": bool(self.show_legend),
            "theme": normalize_plotly_theme(str(self.theme)),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlotlyPlotDisplayOptions:
        """Build display options from a mapping produced by :meth:`to_dict`.

        Unknown keys are ignored and missing keys fall back to field defaults,
        so the widget stays robust across schema evolution.

        Args:
            data: Mapping of option names to values.

        Returns:
            New :class:`PlotlyPlotDisplayOptions` instance.
        """
        known = {field.name for field in fields(cls)}
        kwargs: dict[str, Any] = {
            key: value for key, value in data.items() if key in known
        }
        if "theme" in kwargs:
            kwargs["theme"] = normalize_plotly_theme(str(kwargs["theme"]))
        return cls(**kwargs)
