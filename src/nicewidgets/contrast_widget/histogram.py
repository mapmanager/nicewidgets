"""Histogram plot for contrast widget.

Returns Plotly figure dict (never go.Figure) for ui.plotly / update_figure.

Note: Single source of truth is kymflow.core.plotting.image_plots.histogram_plot_plotly.
Sync or review when kymflow plotting changes.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import plotly.graph_objects as go

from nicewidgets.contrast_widget.theme import ThemeMode, get_theme_colors, get_theme_template


def _resolve_theme(theme: Union[str, ThemeMode]) -> ThemeMode:
    """Convert str to ThemeMode. Default to LIGHT (align with kymflow)."""
    if isinstance(theme, ThemeMode):
        return theme
    s = str(theme).lower()
    if s in ("dark", "plotly_dark"):
        return ThemeMode.DARK
    return ThemeMode.LIGHT


def histogram_plot_plotly(
    image: Optional[np.ndarray],
    zmin: Optional[int] = None,
    zmax: Optional[int] = None,
    log_scale: bool = True,
    theme: Optional[Union[str, ThemeMode]] = None,
    bins: int = 256,
) -> dict:
    """Create a histogram plot of image pixel intensities.

    Args:
        image: 2D numpy array, or None for empty plot
        zmin: Minimum intensity value to show as vertical line (optional)
        zmax: Maximum intensity value to show as vertical line (optional)
        log_scale: If True, use log scale for y-axis (default: True)
        theme: Theme mode (DARK or LIGHT). Defaults to LIGHT if None.
        bins: Number of bins for histogram (default: 256)

    Returns:
        Plotly figure dict ready for ui.plotly / update_figure.
    """
    theme_mode = ThemeMode.LIGHT if theme is None else _resolve_theme(theme)

    template = get_theme_template(theme_mode)
    bg_color, fg_color = get_theme_colors(theme_mode)
    grid_color = "rgba(255,255,255,0.2)" if theme_mode is ThemeMode.DARK else "#cccccc"

    if image is None:
        fig = go.Figure()
        fig.update_layout(
            template=template,
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=fg_color),
        )
        return fig.to_dict()

    flat_image = image.flatten()
    hist, bin_edges = np.histogram(flat_image, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    image_max = float(np.max(flat_image))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=hist,
            marker_color=fg_color,
            opacity=0.7,
        )
    )

    if zmin is not None:
        fig.add_vline(
            x=zmin,
            line_dash="dash",
            line_color="blue",
            line_width=2,
            annotation_text="Min",
            annotation_position="top",
        )

    if zmax is not None:
        fig.add_vline(
            x=zmax,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text="Max",
            annotation_position="top",
        )

    fig.update_layout(
        template=template,
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=fg_color),
        xaxis=dict(
            title="Pixel Intensity",
            color=fg_color,
            gridcolor=grid_color,
            range=[0.0, image_max],
        ),
        yaxis=dict(
            title="Count",
            color=fg_color,
            gridcolor=grid_color,
            type="log" if log_scale else "linear",
        ),
        margin=dict(l=0, r=20, t=10, b=20),
        showlegend=False,
    )

    return fig.to_dict()
