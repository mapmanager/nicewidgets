"""Thin translation helpers between backend responses and Plotly payloads."""

from __future__ import annotations

from dataclasses import dataclass

from nicewidgets.plotly_axis_layout import resolve_plot_layout_margins
from nicewidgets.raster_viewer.backend.image_model import RenderResponse, RowColBounds, ViewRequest, ViewportSize
from nicewidgets.raster_viewer.frontend.plotly_coord_transform import PlotlyCoordTransform

from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

# Plotly built-in sequential scale name (``Gray`` / ``Grays`` are not valid).
DEFAULT_HEATMAP_COLORSCALE: str = 'Greys'

# Plotly accepts a colorscale as a built-in name or a list of ``[stop, color]``
# pairs; both forms are valid for heatmap traces and the PNG encoder.
PlotlyColorscale = str | list[list[float | str]]

# ``responsive: true`` lets the plot resize with the NiceGUI container.
# See https://plotly.com/javascript/configuration-options/
RASTER_VIEWER_PLOTLY_CONFIG: dict[str, object] = {
    # ``False`` suppresses Plotly's ``plotly_doubleclick`` event; ``reset`` lets
    # the custom handler reload the overview PNG after zoom.
    'doubleClick': 'reset',
    'scrollZoom': True,
    'responsive': True,
}


@dataclass(frozen=True)
class PlotlyViewportPayload:
    """Browser-side viewport and relayout payload.

    Attributes:
        relayout: Plotly relayout payload from the browser.
        width_px: Visible plot width in pixels.
        height_px: Visible plot height in pixels.
    """

    relayout: dict[str, object]
    width_px: int
    height_px: int


def build_plotly_figure(
    response: RenderResponse,
    uirevision: str = 'raster-viewer',
    *,
    heatmap_colorscale: PlotlyColorscale | None = None,
) -> dict:
    """Build a Plotly figure dict from a backend render response."""

    # logger.info('RenderResponse:')
    # print(response)
    # logger.info(f'heatmap_colorscale:{heatmap_colorscale}')

    transform = PlotlyCoordTransform(
        nrows=response.shape[0],
        ncols=response.shape[1],
        grid=response.grid,
    )
    x_range = list(transform.row_col_to_plot_x_range(response.bounds))
    y_range = list(transform.row_col_to_plot_y_range(response.bounds))

    xaxis: dict[str, object] = {
        'range': x_range,
        'title': {'text': response.grid.x_unit} if response.grid.x_unit else {},
        'fixedrange': False,
    }
    yaxis: dict[str, object] = {
        'range': y_range,
        'title': {'text': response.grid.y_unit} if response.grid.y_unit else {},
        'fixedrange': False,
    }

    layout: dict[str, object] = {
        'margin': resolve_plot_layout_margins(show_axis_labels=False, show_legend=False),
        'uirevision': uirevision,
        'autosize': True,
        'dragmode': 'zoom',
        'xaxis': xaxis,
        'yaxis': yaxis,
        # Render the trace legend horizontally below the plot instead of the
        # default vertical legend on the right. Harmless when no legend shows.
        'legend': {
            'orientation': 'h',
            'xanchor': 'center',
            'x': 0.5,
            'yanchor': 'top',
            'y': -0.15,
        },
    }

    trace_x0, trace_y0 = _pixel_center_origin(response)

    if response.mode == 'image_png':
        data = [{
            'type': 'image',
            'source': response.png_data_uri,
            'x0': trace_x0,
            'y0': trace_y0,
            'dx': response.dx,
            'dy': response.dy,
        }]

    elif response.mode == 'heatmap_z':
        if response.z is None:
            raise ValueError('Heatmap response requires numeric z data.')

        colorscale = heatmap_colorscale if heatmap_colorscale is not None else DEFAULT_HEATMAP_COLORSCALE
        data = [{
            'type': 'heatmap',
            'z': response.z.tolist(),
            'x0': trace_x0,
            'y0': trace_y0,
            'dx': response.dx,
            'dy': response.dy,
            'zmin': response.zmin,
            'zmax': response.zmax,
            'colorscale': colorscale,
            'showscale': False,
            'hoverongaps': False,
        }]

    else:
        raise ValueError(f'Unsupported render mode: {response.mode}')

    # logger.debug(f'RenderResponse:{response.mode}')

    return {'data': data, 'layout': layout, 'config': dict(RASTER_VIEWER_PLOTLY_CONFIG)}


def _pixel_center_origin(response: RenderResponse) -> tuple[float, float]:
    """Return Plotly raster trace origin as the first pixel center.

    RasterViewService returns ``x0`` and ``y0`` as edge-aligned plot
    coordinates so axis ranges and ROI rectangle shapes can use data extents
    directly. Plotly raster traces place samples at ``x0``/``y0`` with
    ``dx``/``dy`` spacing, so the trace origin must be shifted to the center of
    the first pixel to align rendered pixels with edge-based shapes.
    """
    return response.x0 + response.dx / 2.0, response.y0 + response.dy / 2.0


def _read_axis_range_pair(
    relayout: dict[str, object],
    *,
    axis: str,
    bracket_fallbacks: tuple[float, float],
) -> tuple[float, float]:
    """Read one axis extent from Plotly relayout keys."""
    list_key = f'{axis}.range'
    raw = relayout.get(list_key)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return float(raw[0]), float(raw[1])
    k0 = f'{axis}.range[0]'
    k1 = f'{axis}.range[1]'
    return (
        _get_number(relayout, k0, bracket_fallbacks[0]),
        _get_number(relayout, k1, bracket_fallbacks[1]),
    )


def parse_relayout_payload(
    payload: PlotlyViewportPayload,
    transform: PlotlyCoordTransform,
    fallback_bounds: RowColBounds,
) -> ViewRequest:
    """Convert Plotly relayout payload into a backend view request."""
    relayout = payload.relayout

    fx_lo, fx_hi = transform.row_col_to_plot_x_range(fallback_bounds)
    fy_lo, fy_hi = transform.row_col_to_plot_y_range(fallback_bounds)

    x_lo, x_hi = _read_axis_range_pair(
        relayout,
        axis='xaxis',
        bracket_fallbacks=(fx_lo, fx_hi),
    )
    y_a, y_b = _read_axis_range_pair(
        relayout,
        axis='yaxis',
        bracket_fallbacks=(fy_lo, fy_hi),
    )

    bounds = transform.plot_xy_ranges_to_row_col(x_lo, x_hi, y_a, y_b)
    viewport = ViewportSize(width_px=payload.width_px, height_px=payload.height_px)

    return ViewRequest(bounds=bounds, viewport=viewport)


def _get_number(payload: dict[str, object], key: str, fallback: float) -> float:
    """Read a float value from a relayout payload."""
    value = payload.get(key, fallback)
    return float(value)
