"""Multi-channel raster display: models, RGB composition, and (later) coordinator.

Phase 1 ships contracts + additive RGB composition. The NiceGUI coordinator
:class:`MultiChannelRasterView` lands in a later phase; see package docs and
``examples/raster_viewer``.
"""

from nicewidgets.raster_viewer.multichannel.compose import (
    CompositeChannelLimitError,
    compose_rgb_png_data_uri,
    compose_rgb_uint8,
    rgb_array_to_png_data_uri,
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

__all__ = [
    'CompositeChannelLimitError',
    'ChannelDisplayStyle',
    'ChannelPlane',
    'DEFAULT_CHANNEL_TINTS',
    'MAX_COMPOSITE_CHANNELS',
    'MosaicOrientation',
    'MultiChannelRasterViewConfig',
    'RasterLayoutMode',
    'compose_rgb_png_data_uri',
    'compose_rgb_uint8',
    'default_tint_for_channel',
    'rgb_array_to_png_data_uri',
    'select_composite_planes',
    'validate_same_shape',
]
