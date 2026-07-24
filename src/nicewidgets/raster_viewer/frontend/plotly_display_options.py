"""Display options for the Plotly raster viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nicewidgets.plotly_layout_margins import PlotlyLayoutMarginsProfile
from nicewidgets.plotly_theme import PlotlyThemeName, normalize_plotly_theme

# JSON-serializable scalar fields. ``layout_margins_profile`` is a fixed
# construction-time layout concern, not user-mutable display state, so it is
# intentionally excluded from serialization round trips.
_SERIALIZABLE_FIELDS = (
    'show_plotly_toolbar',
    'show_rois',
    'show_roi_labels',
    'show_trace_overlays',
    'show_x_axis_labels',
    'show_y_axis_labels',
    'show_hover_info',
    'square_plot',
    'theme',
)


@dataclass(slots=True)
class PlotlyRasterViewerDisplayOptions:
    """User-facing display toggles for :class:`PlotlyRasterViewer`.

    Args:
        show_plotly_toolbar: Whether Plotly's modebar is visible.
        show_rois: Whether rectangular ROI overlays are visible.
        show_roi_labels: Whether rectangular ROI overlay labels are visible.
        show_trace_overlays: Whether managed x/y trace overlays are visible.
        show_x_axis_labels: Whether x-axis title text, tick labels, ticks, axis
            lines, and grid lines are visible.
        show_y_axis_labels: Whether y-axis title text, tick labels, ticks, axis
            lines, and grid lines are visible.
        show_hover_info: Whether Plotly emits hover labels for the raster trace.
            Defaults to False to avoid clutter and reduce browser event traffic.
        square_plot: Whether Plotly should constrain the visible raster plot to
            a square plot area.
        theme: Plotly raster viewer color theme.
        layout_margins_profile: Optional fixed margin profile for aligned
            multi-plot stacks.
    """

    show_plotly_toolbar: bool = False
    show_rois: bool = True
    show_roi_labels: bool = True
    show_trace_overlays: bool = True
    show_x_axis_labels: bool = False
    show_y_axis_labels: bool = False
    show_hover_info: bool = False
    square_plot: bool = False
    theme: PlotlyThemeName = 'light'
    layout_margins_profile: PlotlyLayoutMarginsProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of these options.

        ``layout_margins_profile`` is excluded (see module note). ``theme`` is a
        plain string.

        Returns:
            Mapping of serializable option names to values.
        """
        data = {name: getattr(self, name) for name in _SERIALIZABLE_FIELDS}
        data['theme'] = normalize_plotly_theme(str(self.theme))
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlotlyRasterViewerDisplayOptions:
        """Build display options from a mapping produced by :meth:`to_dict`.

        Unknown keys (including ``layout_margins_profile``) are ignored and
        missing keys fall back to field defaults, so the viewer stays robust
        across schema evolution.

        Args:
            data: Mapping of option names to values.

        Returns:
            New :class:`PlotlyRasterViewerDisplayOptions` instance.
        """
        kwargs: dict[str, Any] = {
            name: data[name] for name in _SERIALIZABLE_FIELDS if name in data
        }
        if 'theme' in kwargs:
            kwargs['theme'] = normalize_plotly_theme(str(kwargs['theme']))
        return cls(**kwargs)
