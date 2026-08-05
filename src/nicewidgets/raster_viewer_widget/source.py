"""Public raster data-source protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import RasterDescriptor, RasterPlaneArray, RasterPlaneRequest


@runtime_checkable
class RasterDataSource(Protocol):
    """Provide metadata and synchronous plane access to one widget.

    Implementations remain Python-owned for as long as they are registered by
    a ``RasterViewerWidget``. Methods are synchronous because NiceGUI executes
    route callbacks outside the browser event bridge; implementations that own
    mutable arrays must provide any required thread safety.

    Plane arrays must be two-dimensional, C-contiguous ``uint16`` or
    ``float32`` values. The internal HTTP adapter preserves dtype, emits raw
    little-endian bytes, and never serializes pixels through JSON or NiceGUI
    events. Multidimensional sources interpret the request's named T/Z indices;
    sliding-Z is an optional centered maximum projection at a fixed T.

    The protocol is also the lazy-loading boundary. An implementation may own
    an eager NumPy array, a memory map, a Dask/Zarr-backed array, or a precise
    caller-supplied plane callable. Only the requested 2D plane must be
    materialized before ``get_plane`` returns; the widget never requires the
    complete dataset to be resident in memory.
    """

    def get_descriptor(self) -> RasterDescriptor:
        """Return immutable source metadata and initial committed ROIs.

        Returns:
            Versioned descriptor whose channels share one dtype and shape.
        """
        ...

    def get_plane(self, request: RasterPlaneRequest) -> RasterPlaneArray:
        """Return one contiguous 2D uint16 or float32 plane.

        Args:
            request: Channel identity and optional Z/projection selection.

        Returns:
            C-contiguous plane matching the descriptor's final Y/X dimensions.
        """
        ...
