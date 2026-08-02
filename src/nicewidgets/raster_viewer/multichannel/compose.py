"""Additive RGB composition for multi-channel rasters (v1: <= 3 channels).

Composite coloring follows the acqstore demo pattern: each channel is intensity-
windowed to ``[0, 1]``, mapped through that channel's color LUT to RGB, then
contributions are added and clamped to ``0..255``.
"""

from __future__ import annotations

import base64
import io
import math
from collections.abc import Sequence
from dataclasses import replace
from functools import lru_cache

import numpy as np
from PIL import Image
from plotly.colors import sample_colorscale as plotly_sample_colorscale

from nicewidgets.raster_viewer.backend.image_model import (
    RasterGridSpec,
    RenderResponse,
    RowColBounds,
)
from nicewidgets.raster_viewer.multichannel.models import (
    MAX_COMPOSITE_CHANNELS,
    ChannelPlane,
    default_tint_for_channel,
)

# Soft cap for composite image-trace payloads (JSON HxWx3 can get large).
DEFAULT_COMPOSITE_MAX_PIXELS = 500_000

# Plotly.py reverses these vs Plotly.js for heatmap; match PNG path in RasterViewService.
_PLOTLY_PY_REVERSED_VS_JS: frozenset[str] = frozenset({'Greys', 'Greens', 'Blues'})


class CompositeChannelLimitError(ValueError):
    """Raised when composite RGB would need more than :data:`MAX_COMPOSITE_CHANNELS`."""


def _normalize_plane(
    data: np.ndarray,
    *,
    zmin: float | None,
    zmax: float | None,
) -> np.ndarray:
    """Return a float32 array in ``[0, 1]`` from ``data`` and an intensity window."""
    a = np.asarray(data, dtype=np.float32)
    if a.size == 0:
        return np.zeros(a.shape, dtype=np.float32)

    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.float32)

    lo = float(zmin) if zmin is not None else float(np.nanmin(a))
    hi = float(zmax) if zmax is not None else float(np.nanmax(a))
    if hi <= lo or not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros(a.shape, dtype=np.float32)

    out = np.zeros_like(a, dtype=np.float32)
    np.divide(a - lo, hi - lo, out=out, where=finite)
    np.clip(out, 0.0, 1.0, out=out)
    return np.where(finite, out, 0.0).astype(np.float32, copy=False)


def _is_grayscale_colorscale(colorscale: str | list[list[float | str]]) -> bool:
    """Return True when ``colorscale`` is the default grayscale family."""
    if isinstance(colorscale, str):
        return colorscale in {'Greys', 'Gray', 'grey', 'gray'}
    return False


def _colorscale_cache_key(
    colorscale: str | list[list[float | str]],
) -> tuple[object, ...]:
    """Stable hash key for :func:`_lut_table_uint8`."""
    if isinstance(colorscale, str):
        return ('str', colorscale)
    # Stop list: ((t, color), ...)
    return (
        'list',
        tuple((float(stop[0]), str(stop[1])) for stop in colorscale),
    )


@lru_cache(maxsize=32)
def _lut_table_uint8_cached(key: tuple[object, ...]) -> np.ndarray:
    """Build a ``(256, 3)`` uint8 LUT from a cache key produced by ``_colorscale_cache_key``."""
    kind = key[0]
    if kind == 'str':
        colorscale: str | list[list[float | str]] = str(key[1])
    else:
        colorscale = [[float(t), str(c)] for t, c in key[1]]  # type: ignore[misc]

    stops = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    query = stops
    if isinstance(colorscale, str) and colorscale in _PLOTLY_PY_REVERSED_VS_JS:
        query = 1.0 - stops
    tuples = plotly_sample_colorscale(
        colorscale,
        query.tolist(),
        colortype='tuple',
    )
    lut = np.array(tuples, dtype=np.float64)
    return np.clip(lut * 255.0, 0.0, 255.0).astype(np.uint8)


def lut_table_uint8(colorscale: str | list[list[float | str]]) -> np.ndarray:
    """Return a ``(256, 3)`` uint8 LUT for ``colorscale`` (Plotly-compatible)."""
    return _lut_table_uint8_cached(_colorscale_cache_key(colorscale)).copy()


def validate_same_shape(planes: Sequence[ChannelPlane]) -> tuple[int, int]:
    """Ensure every plane is 2D and shares one ``(rows, cols)`` shape.

    Args:
        planes: Channel planes to validate.

    Returns:
        Shared ``(rows, cols)`` shape.

    Raises:
        ValueError: If ``planes`` is empty or shapes differ.
    """
    if not planes:
        raise ValueError('compose requires at least one channel plane')
    shape = tuple(int(v) for v in np.asarray(planes[0].data).shape)
    if len(shape) != 2:
        raise ValueError(f'channel data must be 2D, got shape={shape}')
    for plane in planes[1:]:
        other = tuple(int(v) for v in np.asarray(plane.data).shape)
        if other != shape:
            raise ValueError(
                f'all channel planes must share the same shape; '
                f'channel {planes[0].channel_id} has {shape}, '
                f'channel {plane.channel_id} has {other}'
            )
    return shape  # type: ignore[return-value]


def select_composite_planes(planes: Sequence[ChannelPlane]) -> list[ChannelPlane]:
    """Return visible planes that will contribute to composite RGB.

    Args:
        planes: All channel planes (visible and hidden).

    Returns:
        Visible planes in input order.

    Raises:
        CompositeChannelLimitError: If more than :data:`MAX_COMPOSITE_CHANNELS`
            visible planes are selected (v1 hard limit).
        ValueError: If no visible planes remain.
    """
    visible = [p for p in planes if p.style.visible]
    if not visible:
        raise ValueError('composite requires at least one visible channel')
    if len(visible) > MAX_COMPOSITE_CHANNELS:
        raise CompositeChannelLimitError(
            f'composite RGB in v1 supports at most {MAX_COMPOSITE_CHANNELS} '
            f'visible channels; got {len(visible)}. Hide channels or use mosaic mode.'
        )
    return visible


def compose_rgb_uint8(planes: Sequence[ChannelPlane]) -> np.ndarray:
    """Compose visible channels into an additive RGB ``uint8`` image.

    For each visible channel:

    1. Normalize with that channel's ``zmin``/``zmax``.
    2. Map through the channel's color LUT (``style.colorscale``), matching the
       acqstore demo ``renderCompositeBitmap`` path.
    3. Add RGB contributions and clamp to ``0..255``.

    Legacy fallback: when ``colorscale`` is grayscale (default ``Greys``) and
    ``tint_rgb`` is set (or defaults via :func:`default_tint_for_channel`), the
    normalized plane is multiplied by the tint instead of sampling a chromatic
    LUT. Explicit ``tint_rgb`` always uses this grayscale×tint path.

    Args:
        planes: Same-shaped channel planes. Hidden channels are skipped.

    Returns:
        Array of shape ``(rows, cols, 3)`` dtype ``uint8``.

    Raises:
        CompositeChannelLimitError: If more than three visible channels.
        ValueError: If shapes differ or no visible channels remain.
    """
    visible = select_composite_planes(planes)
    validate_same_shape(visible)
    rows, cols = visible[0].data.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.float32)

    for plane in visible:
        style = plane.style
        norm = _normalize_plane(plane.data, zmin=style.zmin, zmax=style.zmax)

        use_tint = style.tint_rgb is not None or _is_grayscale_colorscale(style.colorscale)
        if use_tint:
            tint = style.tint_rgb
            if tint is None:
                tint = default_tint_for_channel(plane.channel_id)
            rgb[..., 0] += norm * float(tint[0]) * 255.0
            rgb[..., 1] += norm * float(tint[1]) * 255.0
            rgb[..., 2] += norm * float(tint[2]) * 255.0
            continue

        lut = lut_table_uint8(style.colorscale)
        idx = np.minimum((norm * 255.0).astype(np.int32), 255)
        rgb += lut[idx].astype(np.float32)

    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def rgb_array_to_png_data_uri(rgb: np.ndarray) -> str:
    """Encode an ``(H, W, 3)`` uint8 RGB array as a PNG data URI."""
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'expected RGB array (H, W, 3), got shape={arr.shape}')
    image = Image.fromarray(np.asarray(arr, dtype=np.uint8), mode='RGB')
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def compose_rgb_png_data_uri(planes: Sequence[ChannelPlane]) -> str:
    """Compose visible channels and return a PNG data URI."""
    return rgb_array_to_png_data_uri(compose_rgb_uint8(planes))


def _integer_clip_bounds(data_shape: tuple[int, int], bounds: RowColBounds) -> tuple[int, int, int, int]:
    """Return ``(r0, r1, c0, c1)`` integer slices for ``bounds`` on ``data_shape``."""
    clipped = bounds.clipped_to_shape(data_shape)
    r0 = max(0, int(math.floor(min(clipped.row_min, clipped.row_max))))
    r1 = min(data_shape[0], int(math.ceil(max(clipped.row_min, clipped.row_max))))
    c0 = max(0, int(math.floor(min(clipped.col_min, clipped.col_max))))
    c1 = min(data_shape[1], int(math.ceil(max(clipped.col_min, clipped.col_max))))
    if r1 <= r0:
        r1 = min(data_shape[0], r0 + 1)
    if c1 <= c0:
        c1 = min(data_shape[1], c0 + 1)
    return r0, r1, c0, c1


def compose_rgb_uint8_for_bounds(
    planes: Sequence[ChannelPlane],
    bounds: RowColBounds,
) -> tuple[np.ndarray, RowColBounds]:
    """Compose a row/col RGB clip for ``bounds``.

    Returns:
        ``(rgb, used_bounds)`` where ``rgb`` has shape ``(rows, cols, 3)`` uint8
        in numpy row/column space and ``used_bounds`` is the integer clip extent.
    """
    visible = select_composite_planes(planes)
    shape = validate_same_shape(visible)
    r0, r1, c0, c1 = _integer_clip_bounds(shape, bounds)
    clipped_planes = [
        replace(plane, data=np.asarray(plane.data)[r0:r1, c0:c1])
        for plane in visible
    ]
    rgb = compose_rgb_uint8(clipped_planes)
    used = RowColBounds(
        row_min=float(r0),
        row_max=float(r1),
        col_min=float(c0),
        col_max=float(c1),
    )
    return rgb, used


def downsample_rgb_uint8(
    rgb: np.ndarray,
    *,
    max_pixels: int | None = DEFAULT_COMPOSITE_MAX_PIXELS,
) -> tuple[np.ndarray, int]:
    """Downsample ``(rows, cols, 3)`` RGB with integer stride when over budget.

    Returns:
        ``(rgb_out, stride)`` where ``stride >= 1``.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'expected RGB (H, W, 3), got shape={arr.shape}')
    if max_pixels is None or max_pixels <= 0:
        return arr, 1
    rows, cols = int(arr.shape[0]), int(arr.shape[1])
    if rows * cols <= max_pixels:
        return arr, 1
    stride = int(math.ceil(math.sqrt((rows * cols) / float(max_pixels))))
    stride = max(1, stride)
    return arr[::stride, ::stride], stride


def rgb_rows_cols_to_plotly_image_z(rgb: np.ndarray) -> np.ndarray:
    """Convert numpy ``(rows, cols, 3)`` RGB to Plotly image ``(ncols, nrows, 3)``.

    Matches the heatmap / PNG transpose convention: plot-x along rows, plot-y
    along columns.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'expected RGB (H, W, 3), got shape={arr.shape}')
    return np.transpose(arr, (1, 0, 2))


def build_image_rgb_response(
    planes: Sequence[ChannelPlane],
    *,
    grid: RasterGridSpec,
    bounds: RowColBounds,
    max_pixels: int | None = DEFAULT_COMPOSITE_MAX_PIXELS,
) -> RenderResponse:
    """Compose a viewport clip and return an ``image_rgb`` :class:`RenderResponse`."""
    visible = select_composite_planes(planes)
    shape = validate_same_shape(visible)
    rgb_rc, used = compose_rgb_uint8_for_bounds(visible, bounds)
    rgb_ds, stride = downsample_rgb_uint8(rgb_rc, max_pixels=max_pixels)
    rgb_plotly = rgb_rows_cols_to_plotly_image_z(rgb_ds)
    row0 = float(min(used.row_min, used.row_max))
    col0 = float(min(used.col_min, used.col_max))
    return RenderResponse(
        mode='image_rgb',
        level=0,
        bounds=used,
        shape=shape,
        grid=grid,
        x0=row0 * grid.dx,
        y0=col0 * grid.dy,
        dx=float(stride) * grid.dx,
        dy=float(stride) * grid.dy,
        rgb=np.asarray(rgb_plotly, dtype=np.uint8),
    )
