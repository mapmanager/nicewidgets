"""Opaque source registry and internal binary HTTP transport."""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any
from uuid import uuid4

from fastapi import HTTPException
from nicegui import app
from starlette.responses import Response

from .models import RasterDescriptor, RasterPlaneRequest
from .source import RasterDataSource

LOGGER = logging.getLogger(__name__)
ROUTE_PREFIX = "/_raster_viewer"


class RasterSourceRegistry:
    """Store per-widget Python sources behind unguessable transport tokens."""

    def __init__(self) -> None:
        """Initialize an empty thread-safe registry."""
        self._sources: dict[str, RasterDataSource] = {}
        self._lock = RLock()

    def register(self, source: RasterDataSource) -> str:
        """Register one source and return its opaque token.

        Args:
            source: Protocol-compatible Python raster source.

        Returns:
            Unguessable source token.
        """
        token = uuid4().hex
        with self._lock:
            self._sources[token] = source
        return token

    def unregister(self, token: str) -> None:
        """Remove a source token if it remains registered.

        Args:
            token: Previously issued opaque token.
        """
        with self._lock:
            self._sources.pop(token, None)

    def get(self, token: str) -> RasterDataSource:
        """Return a registered source.

        Args:
            token: Opaque source token.

        Returns:
            Matching source.

        Raises:
            KeyError: If the token is unknown.
        """
        with self._lock:
            return self._sources[token]


REGISTRY = RasterSourceRegistry()
_ROUTES_REGISTERED = False


def _source_or_404(token: str) -> RasterDataSource:
    """Resolve one source token or raise a non-revealing HTTP error."""
    try:
        return REGISTRY.get(token)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="unknown raster source") from error


def _descriptor_json(token: str, descriptor: RasterDescriptor) -> dict[str, Any]:
    """Build the versioned JavaScript descriptor and binary-plane contract.

    Channel endpoints return one little-endian, C-contiguous ``(Y, X)`` plane
    with no container header. The descriptor declares the exact dtype,
    encoding, dimensions, and byte count. JavaScript validates these fields,
    decodes the response into a typed array, then applies the documented
    transpose-plus-flip-Y display transform.

    Args:
        token: Opaque registry token used to construct plane URLs.
        descriptor: Validated source descriptor.

    Returns:
        Canonical snake-case JSON object understood by the browser viewer.
    """
    height, width = descriptor.header.shape[-2:]
    y_step, x_step = descriptor.header.physical_units[-2:]
    y_label, x_label = descriptor.header.physical_units_labels[-2:]
    item_size = 2 if descriptor.header.dtype == "uint16" else 4
    return {
        "schema_version": descriptor.schema_version,
        "id": descriptor.source_id,
        "label": descriptor.label,
        "header": descriptor.header.to_json(),
        "width": width,
        "height": height,
        "layout": "row-major",
        "endianness": "little",
        "display_orientation": {"transpose": True, "flip_y": True},
        "axes": {
            "x": {"label": x_label, "step": x_step, "unit": ""},
            "y": {"label": y_label, "step": y_step, "unit": ""},
        },
        "rois": list(descriptor.rois),
        "channels": [
            {
                "id": channel.channel_id,
                "index": index,
                "label": channel.label,
                "dtype": descriptor.header.dtype,
                "encoding": (
                    "raw-u16-le" if descriptor.header.dtype == "uint16" else "raw-f32-le"
                ),
                "byte_length": height * width * item_size,
                "display": channel.display.to_json(),
                "data_url": (
                    f"{ROUTE_PREFIX}/{token}/channels/{channel.channel_id}/plane"
                ),
            }
            for index, channel in enumerate(descriptor.channels)
        ],
    }


def source_descriptor(token: str) -> dict[str, Any]:
    """Serve browser metadata for a registered source."""
    source = _source_or_404(token)
    return _descriptor_json(token, source.get_descriptor())


def source_plane(
    token: str,
    channel_id: str,
    t_index: int | None = None,
    z_index: int | None = None,
    plus_minus_z: int = 0,
) -> Response:
    """Serve one raw, identity-encoded scalar plane.

    The response is C-order little-endian binary with no container header.
    ``Content-Encoding: identity`` deliberately bypasses NiceGUI's default
    high-level gzip compression, which otherwise buffers large, compressible
    planes before sending response headers. ``Cache-Control: no-store`` keeps
    ephemeral Python source state out of HTTP caches; the viewer owns its
    dataset-scoped decoded-plane cache.

    Args:
        token: Opaque source registration token.
        channel_id: Dataset-local scalar channel identity.
        t_index: Optional zero-based T index.
        z_index: Optional zero-based Z center.
        plus_minus_z: Non-negative centered maximum-projection radius.

    Returns:
        Octet-stream response whose byte count matches descriptor metadata.

    Raises:
        HTTPException: If the token, channel, plane, or projection is invalid.
    """
    source = _source_or_404(token)
    try:
        plane = source.get_plane(
            RasterPlaneRequest(channel_id, t_index, z_index, plus_minus_z)
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail="unknown raster channel") from error
    except (IndexError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    body = plane.tobytes(order="C")
    LOGGER.info(
        "Serving widget plane %s/%s t=%s z=%s radius=%d (%d bytes)",
        token[:8],
        channel_id,
        t_index,
        z_index,
        plus_minus_z,
        len(body),
    )
    height, width = (int(plane.shape[0]), int(plane.shape[1])) if plane.ndim == 2 else (-1, -1)
    itemsize = int(plane.dtype.itemsize)
    expected_bytes = height * width * itemsize if height >= 0 else -1
    if plane.dtype.kind == "u" and itemsize == 2:
        transport_dtype = "uint16"
        encoding = "raw-u16-le"
    elif plane.dtype.kind == "f" and itemsize == 4:
        transport_dtype = "float32"
        encoding = "raw-f32-le"
    else:
        transport_dtype = str(plane.dtype)
        encoding = f"raw-{transport_dtype}"
    LOGGER.info(
        "Widget plane transport %s/%s dtype=%s encoding=%s shape=%sx%s "
        "itemsize=%d expected_bytes=%d body_bytes=%d match=%s",
        token[:8],
        channel_id,
        transport_dtype,
        encoding,
        height,
        width,
        itemsize,
        expected_bytes,
        len(body),
        expected_bytes == len(body),
    )
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers={
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
            "Content-Encoding": "identity",
        },
    )


def ensure_routes_registered() -> None:
    """Register internal widget routes exactly once per Python process."""
    global _ROUTES_REGISTERED  # noqa: PLW0603
    if _ROUTES_REGISTERED:
        return
    app.add_api_route(f"{ROUTE_PREFIX}/{{token}}/descriptor", source_descriptor, methods=["GET"])
    app.add_api_route(
        f"{ROUTE_PREFIX}/{{token}}/channels/{{channel_id}}/plane",
        source_plane,
        methods=["GET"],
    )
    _ROUTES_REGISTERED = True
