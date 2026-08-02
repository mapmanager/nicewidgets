"""Additive RGB composition for multi-channel rasters (v1: <= 3 channels)."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence

import numpy as np
from PIL import Image

from nicewidgets.raster_viewer.multichannel.models import (
    MAX_COMPOSITE_CHANNELS,
    ChannelPlane,
    default_tint_for_channel,
)


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

    Each visible channel is normalized with its own ``zmin``/``zmax``, multiplied
    by its ``tint_rgb`` (or :func:`default_tint_for_channel` when tint is
    ``None``), then added. Values are clamped to ``0..255``.

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
        tint = style.tint_rgb
        if tint is None:
            tint = default_tint_for_channel(plane.channel_id)
        norm = _normalize_plane(plane.data, zmin=style.zmin, zmax=style.zmax)
        rgb[..., 0] += norm * float(tint[0])
        rgb[..., 1] += norm * float(tint[1])
        rgb[..., 2] += norm * float(tint[2])

    return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


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
