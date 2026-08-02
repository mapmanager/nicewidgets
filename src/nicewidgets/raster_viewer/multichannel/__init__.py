"""Multi-channel raster display: models, RGB composition, and coordinator.

- Phase 1: contracts + additive RGB composition
- Phase 2: :class:`MultiChannelRasterView` (single / mosaic + linked viewport)
- Phase 3: composite RGB pane via Plotly ``image`` HxWx3 (not PNG)

``MultiChannelRasterView`` is imported lazily so pure backend helpers (compose /
models) can be tested without pulling NiceGUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nicewidgets.raster_viewer.multichannel.compose import (
    DEFAULT_COMPOSITE_MAX_PIXELS,
    CompositeChannelLimitError,
    build_image_rgb_response,
    compose_rgb_png_data_uri,
    compose_rgb_uint8,
    compose_rgb_uint8_for_bounds,
    downsample_rgb_uint8,
    rgb_array_to_png_data_uri,
    rgb_rows_cols_to_plotly_image_z,
    select_composite_planes,
    validate_same_shape,
)
from nicewidgets.raster_viewer.multichannel.models import (
    DEFAULT_CHANNEL_TINTS,
    MAX_COMPOSITE_CHANNELS,
    ChannelDisplayStyle,
    ChannelPlane,
    MosaicOrientation,
    MultiChannelRasterViewConfig,
    RasterLayoutMode,
    default_tint_for_channel,
)

if TYPE_CHECKING:
    from nicewidgets.raster_viewer.multichannel.view import MultiChannelRasterView as MultiChannelRasterView

__all__ = [
    'COMPOSITE_VIEWER_KEY',
    'CompositeChannelLimitError',
    'ChannelDisplayStyle',
    'ChannelPlane',
    'DEFAULT_CHANNEL_TINTS',
    'DEFAULT_COMPOSITE_MAX_PIXELS',
    'MAX_COMPOSITE_CHANNELS',
    'MosaicOrientation',
    'MultiChannelRasterView',
    'MultiChannelRasterViewConfig',
    'RasterLayoutMode',
    'build_image_rgb_response',
    'compose_rgb_png_data_uri',
    'compose_rgb_uint8',
    'compose_rgb_uint8_for_bounds',
    'default_tint_for_channel',
    'downsample_rgb_uint8',
    'rgb_array_to_png_data_uri',
    'rgb_rows_cols_to_plotly_image_z',
    'select_composite_planes',
    'validate_same_shape',
]


def __getattr__(name: str) -> Any:
    if name == 'MultiChannelRasterView':
        from nicewidgets.raster_viewer.multichannel.view import MultiChannelRasterView

        return MultiChannelRasterView
    if name == 'COMPOSITE_VIEWER_KEY':
        from nicewidgets.raster_viewer.multichannel.view import COMPOSITE_VIEWER_KEY

        return COMPOSITE_VIEWER_KEY
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
