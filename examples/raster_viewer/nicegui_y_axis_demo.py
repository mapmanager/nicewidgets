"""Spike demo for Ticket 022: verify Plotly y-axis origin behavior.

Builds a 2x2 grid of small Plotly viewers using the **same transposed image
data** but with different ``yaxis.range`` directions and trace modes:

1. ``image_png`` + top-down range ``[N, 0]``.
2. ``image_png`` + bottom-up range ``[0, N]``.
3. ``heatmap_z`` + top-down range ``[N, 0]``.
4. ``heatmap_z`` + bottom-up range ``[0, N]``.

The raw image marks each raw-coordinate corner with a unique block value so the
``T[0, 0]`` corner (which equals ``raw[0, 0]`` after the production transpose)
is identifiable. Under the bottom-up y axis, that corner should map to the
bottom-left of every plot. Note that Plotly heatmap and the PNG path can show
different visual polarity for the ``Greys`` colorscale, so use the distinct
corner pattern rather than whiteness alone when comparing cells.

Run with::

    uv run python src/nicewidgets/raster_viewer/demo/nicegui_y_axis_demo.py

Purpose: confirm whether Plotly's ``image`` trace honors a bottom-up y axis
the same way ``heatmap`` does. The answer drives the strategy for the PNG path
in the Ticket 022 refactor (no change vs pre-flip vs negative dy).

This script does not import from ``cloudscope`` and does not depend on the
production raster viewer's transform; it builds minimal Plotly figure dicts
directly so each scenario is explicit.
"""

from __future__ import annotations

import numpy as np
from nicegui import ui

from nicewidgets.raster_viewer.backend.image_model import RasterDisplayStyle
from nicewidgets.raster_viewer.backend.raster_service import RasterViewService


def _make_asymmetric_raw(size: int = 16) -> np.ndarray:
    """Build raw[rows=size, cols=size] with unique corner block values.

    Args:
        size: Side length of the square raw image.

    Returns:
        Float32 image with ``raw[0, 0]`` corner brightest, distinct blocks at
        every corner, and a mid-tone background.
    """
    data = np.full((size, size), 0.5, dtype=np.float32)
    data[0:3, 0:3] = 1.00      # bright    -> "raw top-left (row=0, col=0)"
    data[0:3, -3:] = 0.75      # mid-light -> "raw top-right (row=0, col=N-1)"
    data[-3:, 0:3] = 0.25      # mid-dark  -> "raw bottom-left (row=N-1, col=0)"
    data[-3:, -3:] = 0.00      # darkest   -> "raw bottom-right (row=N-1, col=N-1)"
    return data


def _build_figure(
    *,
    mode: str,
    y_range: list[float],
    title: str,
    t_arr: np.ndarray,
) -> dict[str, object]:
    """Build a minimal Plotly figure dict for one spike cell.

    Args:
        mode: ``image`` or ``heatmap``.
        y_range: Plotly ``layout.yaxis.range`` for this cell.
        title: Figure title shown above the cell.
        t_arr: Transposed image array passed to the trace.

    Returns:
        Plotly figure dictionary.
    """
    ny, nx = t_arr.shape  # T shape = (cols_raw, rows_raw)
    layout: dict[str, object] = {
        'title': {'text': title, 'font': {'size': 12}},
        'margin': {'l': 50, 'r': 20, 't': 60, 'b': 40},
        'xaxis': {
            'range': [0.0, float(nx)],
            'title': {'text': 'x (raw row index)'},
            'fixedrange': True,
        },
        'yaxis': {
            'range': y_range,
            'title': {'text': 'y (raw col index)'},
            'fixedrange': True,
        },
        'autosize': True,
    }

    if mode == 'image':
        style = RasterDisplayStyle(colorscale='Greys', zmin=0.0, zmax=1.0)
        data: list[dict[str, object]] = [{
            'type': 'image',
            'source': RasterViewService.array_to_png_data_uri(t_arr, style=style),
            'x0': 0.0,
            'y0': 0.0,
            'dx': 1.0,
            'dy': 1.0,
        }]
        layout['yaxis']['scaleanchor'] = False
    elif mode == 'heatmap':
        data = [{
            'type': 'heatmap',
            'z': t_arr.tolist(),
            'x0': 0.0,
            'y0': 0.0,
            'dx': 1.0,
            'dy': 1.0,
            'colorscale': 'Greys',
            'zmin': 0.0,
            'zmax': 1.0,
            'showscale': False,
            'reversescale': False,
        }]
    else:
        raise ValueError(f'Unknown spike mode: {mode}')

    return {'data': data, 'layout': layout}


@ui.page('/')
def home() -> None:
    """Render four spike cells for visual y-axis-orientation comparison."""
    raw = _make_asymmetric_raw(size=16)
    t_arr = raw.T  # mirrors production transpose in raster_service.render
    ny, _ = t_arr.shape
    y_lo = 0.0
    y_hi = float(ny)

    legend = (
        'Raw image marks each corner with a unique block value:\n'
        '  raw[0:3, 0:3]   = 1.00  (BRIGHTEST)\n'
        '  raw[0:3, -3:]   = 0.75\n'
        '  raw[-3:, 0:3]   = 0.25\n'
        '  raw[-3:, -3:]   = 0.00  (BLACK)\n'
        '\n'
        'After production transpose, the BRIGHTEST block lives at T[0:3, 0:3].\n'
        'Under the bottom-up y axis, that coordinate block should appear at the\n'
        'BOTTOM-LEFT of every plot. PNG and heatmap can differ in Greys polarity.'
    )

    with ui.column().classes('w-full gap-4 p-4'):
        ui.label('Ticket 022 spike — y-axis origin behavior').classes('text-lg font-semibold')
        ui.label(legend).classes('whitespace-pre-line text-xs font-mono')

        with ui.grid(columns=2).classes('w-full gap-4'):
            ui.plotly(_build_figure(
                mode='image',
                y_range=[y_hi, y_lo],
                title='IMAGE PNG + top-down [y_hi, y_lo]',
                t_arr=t_arr,
            )).classes('w-full h-[420px]')
            ui.plotly(_build_figure(
                mode='image',
                y_range=[y_lo, y_hi],
                title='IMAGE PNG + bottom-up [y_lo, y_hi]',
                t_arr=t_arr,
            )).classes('w-full h-[420px]')
            ui.plotly(_build_figure(
                mode='heatmap',
                y_range=[y_hi, y_lo],
                title='HEATMAP + top-down [y_hi, y_lo]',
                t_arr=t_arr,
            )).classes('w-full h-[420px]')
            ui.plotly(_build_figure(
                mode='heatmap',
                y_range=[y_lo, y_hi],
                title='HEATMAP + bottom-up [y_lo, y_hi]',
                t_arr=t_arr,
            )).classes('w-full h-[420px]')


if __name__ in {'__main__', '__mp_main__'}:
    ui.run(reload=False, native=True)
