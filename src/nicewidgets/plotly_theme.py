"""Shared Plotly light/dark layout theme helpers for nicewidgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PlotlyThemeName = Literal['light', 'dark']


@dataclass(frozen=True, slots=True)
class PlotlyTheme:
    """Colors used to render a Plotly light/dark layout theme.

    Args:
        paper_bgcolor: Color for the Plotly paper/background area.
        plot_bgcolor: Color for the plotting area.
        font_color: Default text color.
        axis_color: Axis title, tick, and line color.
        grid_color: Axis grid color.
        zero_line_color: Axis zero-line color.
    """

    paper_bgcolor: str
    plot_bgcolor: str
    font_color: str
    axis_color: str
    grid_color: str
    zero_line_color: str


PLOTLY_THEMES: dict[PlotlyThemeName, PlotlyTheme] = {
    'light': PlotlyTheme(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font_color='#111827',
        axis_color='#374151',
        grid_color='#e5e7eb',
        zero_line_color='#d1d5db',
    ),
    'dark': PlotlyTheme(
        paper_bgcolor='#111827',
        plot_bgcolor='#111827',
        font_color='#f9fafb',
        axis_color='#d1d5db',
        grid_color='#374151',
        zero_line_color='#4b5563',
    ),
}


def normalize_plotly_theme(value: str) -> PlotlyThemeName:
    """Return a supported Plotly theme name.

    Args:
        value: Candidate theme name.

    Returns:
        ``'dark'`` when ``value`` is dark; otherwise ``'light'``.
    """
    return 'dark' if str(value).lower() == 'dark' else 'light'


def theme_for_name(name: PlotlyThemeName) -> PlotlyTheme:
    """Return the color palette for a supported theme name.

    Args:
        name: Supported theme name.

    Returns:
        Theme color palette.
    """
    return PLOTLY_THEMES[name]


def apply_plotly_theme_to_layout(layout: dict[str, object], name: PlotlyThemeName) -> None:
    """Apply layout/axis colors for a Plotly theme to a figure layout dict.

    Args:
        layout: Plotly figure layout dictionary to mutate in place.
        name: Supported theme name.

    Returns:
        None.
    """
    theme = theme_for_name(name)
    layout['paper_bgcolor'] = theme.paper_bgcolor
    layout['plot_bgcolor'] = theme.plot_bgcolor
    layout['font'] = {'color': theme.font_color}

    for axis_name in ('xaxis', 'yaxis'):
        axis = layout.setdefault(axis_name, {})
        if not isinstance(axis, dict):
            axis = {}
            layout[axis_name] = axis
        axis['color'] = theme.axis_color
        axis['linecolor'] = theme.axis_color
        axis['tickcolor'] = theme.axis_color
        axis['gridcolor'] = theme.grid_color
        axis['zerolinecolor'] = theme.zero_line_color
