"""Contracts for multi-channel raster display (mosaic + composite).

These models are intentionally UI-agnostic so the composite encoder and the
future :class:`MultiChannelRasterView` coordinator share one vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# Composite RGB in v1 is limited to three contributing channels (R/G/B mapping).
# Mosaic / single-channel paths may still carry more channels; only composition
# raises when this limit is exceeded.
MAX_COMPOSITE_CHANNELS = 3

# Default additive tints for channels 0..2 when the host does not assign colors.
DEFAULT_CHANNEL_TINTS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

RasterLayoutMode = Literal['single', 'mosaic', 'composite']
MosaicOrientation = Literal['horizontal', 'vertical']


@dataclass(frozen=True, slots=True)
class ChannelDisplayStyle:
    """Per-channel contrast and color contribution.

    Args:
        visible: Whether the channel contributes to mosaic and/or composite.
        zmin: Lower intensity window; ``None`` uses the plane min.
        zmax: Upper intensity window; ``None`` uses the plane max.
        colorscale: Plotly colorscale / LUT for single/mosaic panes and for
            composite RGB sampling (intensity → RGB, then additive merge).
        tint_rgb: Optional additive RGB tint in ``0..1``. When set, composite
            uses grayscale×tint instead of sampling ``colorscale``. When
            ``None`` and ``colorscale`` is grayscale, compose falls back to
            :func:`default_tint_for_channel`. Ignored when the channel is not
            visible.
    """

    visible: bool = True
    zmin: float | None = None
    zmax: float | None = None
    colorscale: str | list[list[float | str]] = 'Greys'
    tint_rgb: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.tint_rgb is None:
            return
        r, g, b = self.tint_rgb
        for name, value in (('r', r), ('g', g), ('b', b)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f'tint_rgb {name} must be in [0, 1], got {value}')


@dataclass(frozen=True, slots=True)
class ChannelPlane:
    """One same-shaped channel plane plus display style.

    Args:
        channel_id: Stable zero-based channel index (host identity).
        label: Optional UI label; defaults to ``str(channel_id)`` at the host.
        data: Full-resolution 2D array ``(rows, columns)``.
        style: Contrast / tint / visibility for this channel.
    """

    channel_id: int
    data: np.ndarray
    style: ChannelDisplayStyle = ChannelDisplayStyle()
    label: str | None = None

    def __post_init__(self) -> None:
        if int(self.channel_id) < 0:
            raise ValueError(f'channel_id must be >= 0, got {self.channel_id}')
        arr = np.asarray(self.data)
        if arr.ndim != 2:
            raise ValueError(f'channel data must be 2D, got shape={arr.shape}')
        object.__setattr__(self, 'data', arr)


@dataclass(frozen=True, slots=True)
class MultiChannelRasterViewConfig:
    """Coordinator display policy (constructor defaults + runtime setters).

    Args:
        layout_mode: ``single`` (one pane), ``mosaic`` (all visible channels),
            or ``composite`` (RGB merge; v1 requires <= 3 visible channels).
        mosaic_orientation: ``horizontal`` → 1×N grid; ``vertical`` → N×1.
        link_viewport: When ``True``, pan/zoom stays synchronized across panes.
    """

    layout_mode: RasterLayoutMode = 'single'
    mosaic_orientation: MosaicOrientation = 'horizontal'
    link_viewport: bool = True


def default_tint_for_channel(channel_id: int) -> tuple[float, float, float]:
    """Return the default R/G/B tint for ``channel_id`` (cycles after 2)."""
    return DEFAULT_CHANNEL_TINTS[int(channel_id) % len(DEFAULT_CHANNEL_TINTS)]
