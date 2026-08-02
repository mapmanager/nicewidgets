"""Multi-channel raster display: models, RGB composition, and coordinator.

- Phase 1: contracts + additive RGB composition
- Phase 2: :class:`MultiChannelRasterView` (single / mosaic + linked viewport)
- Phase 3: composite RGB pane (helpers already available)

``MultiChannelRasterView`` is imported lazily so pure backend helpers (compose /
models) can be tested without pulling NiceGUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from nicewidgets.raster_viewer.multichannel.view import MultiChannelRasterView as MultiChannelRasterView

__all__ = [
    'CompositeChannelLimitError',
    'ChannelDisplayStyle',
    'ChannelPlane',
    'DEFAULT_CHANNEL_TINTS',
    'MAX_COMPOSITE_CHANNELS',
    'MosaicOrientation',
    'MultiChannelRasterView',
    'MultiChannelRasterViewConfig',
    'RasterLayoutMode',
    'compose_rgb_png_data_uri',
    'compose_rgb_uint8',
    'default_tint_for_channel',
    'rgb_array_to_png_data_uri',
    'select_composite_planes',
    'validate_same_shape',
]


def __getattr__(name: str) -> Any:
    if name == 'MultiChannelRasterView':
        from nicewidgets.raster_viewer.multichannel.view import MultiChannelRasterView

        return MultiChannelRasterView
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
