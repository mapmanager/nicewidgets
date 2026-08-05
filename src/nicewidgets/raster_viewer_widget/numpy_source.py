"""In-memory NumPy implementation of the raster data-source protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt

from .models import (
    RasterChannelDescriptor,
    RasterChannelDisplay,
    RasterDescriptor,
    RasterHeader,
    RasterPlaneArray,
    RasterPlaneRequest,
)
from .roi import LineRoi, RectRoi, Roi

type RasterArray = npt.NDArray[np.uint16] | npt.NDArray[np.float32]
type RoiInput = Roi | Mapping[str, object]


def _serialize_rois(rois: Sequence[RoiInput]) -> tuple[dict[str, object], ...]:
    """Serialize typed ROIs while preserving descriptor-boundary mappings."""
    return tuple(
        roi.to_json() if isinstance(roi, (RectRoi, LineRoi)) else dict(roi)
        for roi in rois
    )


def _normalize_array(values: npt.NDArray[np.generic]) -> RasterArray:
    """Return a supported little-endian contiguous array.

    Args:
        values: Source NumPy array.

    Returns:
        Contiguous uint16 or float32 array.

    Raises:
        TypeError: If the array dtype is unsupported.
    """
    if values.dtype == np.dtype(np.uint16):
        return np.ascontiguousarray(values, dtype=np.dtype("<u2"))
    if values.dtype == np.dtype(np.float32):
        return np.ascontiguousarray(values, dtype=np.dtype("<f4"))
    raise TypeError("raster arrays must use uint16 or float32")


@dataclass(frozen=True, slots=True)
class _ChannelView:
    """Pair one channel descriptor with its logical scalar array."""

    descriptor: RasterChannelDescriptor
    values: RasterArray


class NumPyRasterSource:
    """Expose one or more same-shaped NumPy channels as raster planes."""

    def __init__(self, descriptor: RasterDescriptor, channels: tuple[_ChannelView, ...]) -> None:
        """Initialize a validated in-memory source.

        Args:
            descriptor: Browser-facing source metadata.
            channels: Logical scalar channel arrays.
        """
        self._descriptor = descriptor
        self._channels = {channel.descriptor.channel_id: channel for channel in channels}

    @classmethod
    def from_channels(
        cls,
        channels: Sequence[npt.NDArray[np.generic]] | Mapping[str, npt.NDArray[np.generic]],
        *,
        dims: Sequence[str],
        physical_units: Sequence[float],
        physical_units_labels: Sequence[str],
        rois: Sequence[RoiInput] = (),
        source_id: str | None = None,
        label: str = "NumPy raster",
        default_luts: Sequence[str] | None = None,
        channel_displays: Sequence[RasterChannelDisplay] | None = None,
    ) -> NumPyRasterSource:
        """Create a source from separate same-shaped channel arrays.

        Args:
            channels: Ordered arrays or channel-ID-to-array mapping.
            dims: Per-channel dimension labels, such as ``("Z", "Y", "X")``.
            physical_units: Sample spacing corresponding to ``dims``.
            physical_units_labels: Unit labels corresponding to ``dims``.
            rois: Initial AcqStore-compatible ROI envelopes.
            source_id: Optional stable source ID; a UUID is generated when omitted.
            label: Human-readable dataset label.
            default_luts: Optional LUT name for each channel.
            channel_displays: Optional complete initial display state per channel.

        Returns:
            Validated in-memory raster source.

        Raises:
            ValueError: If arrays are absent, differ in shape or dtype, or metadata
                does not describe them.
        """
        if isinstance(channels, Mapping):
            items = list(channels.items())
        else:
            items = [(f"channel_{index}", values) for index, values in enumerate(channels)]
        if not items:
            raise ValueError("at least one channel is required")
        normalized = [(channel_id, _normalize_array(values)) for channel_id, values in items]
        first_shape = normalized[0][1].shape
        first_dtype = normalized[0][1].dtype
        if any(values.shape != first_shape for _, values in normalized):
            raise ValueError("all channels must have the same shape")
        if any(values.dtype != first_dtype for _, values in normalized):
            raise ValueError("all channels must have the same dtype")
        canonical_dims = tuple(dims)
        if len(canonical_dims) != len(first_shape):
            raise ValueError("dims must describe every per-channel array axis")
        if default_luts is None:
            palette = ("gray", "green", "magenta", "cyan", "red", "blue")
            luts = tuple(palette[index % len(palette)] for index in range(len(normalized)))
        else:
            luts = tuple(default_luts)
            if len(luts) != len(normalized):
                raise ValueError("default_luts must match the channel count")
        if channel_displays is not None and default_luts is not None:
            raise ValueError("provide default_luts or channel_displays, not both")
        displays = (
            tuple(channel_displays)
            if channel_displays is not None
            else tuple(RasterChannelDisplay(lut=lut) for lut in luts)
        )
        if len(displays) != len(normalized):
            raise ValueError("channel_displays must match the channel count")
        channel_views = tuple(
            _ChannelView(
                RasterChannelDescriptor(channel_id, channel_id, displays[index]),
                values,
            )
            for index, (channel_id, values) in enumerate(normalized)
        )
        dtype = "uint16" if first_dtype.kind == "u" else "float32"
        header = RasterHeader(
            shape=tuple(int(size) for size in first_shape),
            dims=canonical_dims,
            dtype=dtype,
            num_channels=len(channel_views),
            physical_units=tuple(float(value) for value in physical_units),
            physical_units_labels=tuple(physical_units_labels),
        )
        descriptor = RasterDescriptor(
            source_id=source_id or uuid4().hex,
            label=label,
            header=header,
            channels=tuple(channel.descriptor for channel in channel_views),
            rois=_serialize_rois(rois),
        )
        return cls(descriptor, channel_views)

    @classmethod
    def from_array(
        cls,
        data: npt.NDArray[np.generic],
        *,
        dims: Sequence[str],
        physical_units: Sequence[float],
        physical_units_labels: Sequence[str],
        rois: Sequence[RoiInput] = (),
        source_id: str | None = None,
        label: str = "NumPy raster",
        channel_ids: Sequence[str] | None = None,
        default_luts: Sequence[str] | None = None,
        channel_displays: Sequence[RasterChannelDisplay] | None = None,
    ) -> NumPyRasterSource:
        """Create a source from one explicitly dimensioned NumPy array.

        A named ``C`` axis is split into logical channel views without copying
        the full dataset. The browser descriptor excludes that channel axis.

        Args:
            data: uint16 or float32 array containing all dimensions.
            dims: Axis labels including an optional ``C`` dimension.
            physical_units: Sample spacing corresponding to input ``dims``.
            physical_units_labels: Unit labels corresponding to input ``dims``.
            rois: Initial AcqStore-compatible ROI envelopes.
            source_id: Optional stable source ID.
            label: Human-readable dataset label.
            channel_ids: Optional IDs matching the C-axis length.
            default_luts: Optional LUT names matching the channel count.
            channel_displays: Optional complete initial display state per channel.

        Returns:
            Validated in-memory raster source.

        Raises:
            ValueError: If axis metadata is ambiguous or unsupported.
        """
        canonical_dims = tuple(dims)
        if len(canonical_dims) != data.ndim:
            raise ValueError("dims must describe every input array axis")
        if len(set(canonical_dims)) != len(canonical_dims):
            raise ValueError("dimension names must be unique")
        if canonical_dims.count("C") > 1:
            raise ValueError("at most one C dimension is supported")
        if not (len(physical_units) == len(physical_units_labels) == data.ndim):
            raise ValueError("physical metadata must describe every input array axis")
        normalized = _normalize_array(data)
        if "C" not in canonical_dims:
            if channel_ids is not None and len(channel_ids) != 1:
                raise ValueError("an array without C has exactly one logical channel")
            identifier = channel_ids[0] if channel_ids else "channel_0"
            return cls.from_channels(
                {identifier: normalized},
                dims=canonical_dims,
                physical_units=physical_units,
                physical_units_labels=physical_units_labels,
                rois=rois,
                source_id=source_id,
                label=label,
                default_luts=default_luts,
                channel_displays=channel_displays,
            )
        channel_axis = canonical_dims.index("C")
        channel_count = normalized.shape[channel_axis]
        identifiers = (
            tuple(channel_ids)
            if channel_ids is not None
            else tuple(f"channel_{index}" for index in range(channel_count))
        )
        if len(identifiers) != channel_count:
            raise ValueError("channel_ids must match the C-axis length")
        per_channel_dims = tuple(dim for dim in canonical_dims if dim != "C")
        per_channel_units = tuple(
            value for index, value in enumerate(physical_units) if index != channel_axis
        )
        per_channel_labels = tuple(
            value for index, value in enumerate(physical_units_labels) if index != channel_axis
        )
        if default_luts is None:
            palette = ("gray", "green", "magenta", "cyan", "red", "blue")
            luts = tuple(palette[index % len(palette)] for index in range(channel_count))
        else:
            luts = tuple(default_luts)
            if len(luts) != channel_count:
                raise ValueError("default_luts must match the C-axis length")
        channel_first = np.moveaxis(normalized, channel_axis, 0)
        if channel_displays is not None and default_luts is not None:
            raise ValueError("provide default_luts or channel_displays, not both")
        displays = (
            tuple(channel_displays)
            if channel_displays is not None
            else tuple(RasterChannelDisplay(lut=lut) for lut in luts)
        )
        if len(displays) != channel_count:
            raise ValueError("channel_displays must match the C-axis length")
        channel_views = tuple(
            _ChannelView(
                RasterChannelDescriptor(identifier, identifier, displays[index]),
                cast(RasterArray, channel_first[index]),
            )
            for index, identifier in enumerate(identifiers)
        )
        dtype = "uint16" if normalized.dtype.kind == "u" else "float32"
        per_channel_shape = tuple(
            int(size) for index, size in enumerate(normalized.shape) if index != channel_axis
        )
        header = RasterHeader(
            shape=per_channel_shape,
            dims=per_channel_dims,
            dtype=dtype,
            num_channels=channel_count,
            physical_units=tuple(float(value) for value in per_channel_units),
            physical_units_labels=per_channel_labels,
        )
        descriptor = RasterDescriptor(
            source_id=source_id or uuid4().hex,
            label=label,
            header=header,
            channels=tuple(channel.descriptor for channel in channel_views),
            rois=_serialize_rois(rois),
        )
        return cls(descriptor, channel_views)

    def get_descriptor(self) -> RasterDescriptor:
        """Return source metadata and initial ROIs."""
        return self._descriptor

    def get_plane(self, request: RasterPlaneRequest) -> RasterPlaneArray:
        """Return one 2D plane or centered maximum projection.

        Args:
            request: Channel and optional Z-plane selection.

        Returns:
            Contiguous 2D array retaining source dtype.
        """
        try:
            values = self._channels[request.channel_id].values
        except KeyError as error:
            raise KeyError(f"unknown channel: {request.channel_id}") from error
        radius = request.plus_minus_z
        if radius < 0:
            raise ValueError("plus_minus_z must be non-negative")
        dims = self._descriptor.header.dims
        selections = {"T": request.t_index, "Z": request.z_index}
        indexer: list[int | slice] = []
        projection_axis: int | None = None
        for axis, dim in enumerate(dims[:-2]):
            selected = selections[dim]
            if selected is None:
                raise ValueError(f"{dim} sources require {dim.lower()}_index")
            if not 0 <= selected < values.shape[axis]:
                raise IndexError(f"{dim.lower()}_index is outside the source extent")
            if dim == "Z" and radius:
                indexer.append(
                    slice(
                        max(0, selected - radius),
                        min(values.shape[axis], selected + radius + 1),
                    )
                )
                projection_axis = axis
            else:
                indexer.append(selected)
        for dim, selected in selections.items():
            if dim not in dims and selected is not None:
                raise ValueError(f"source has no {dim} dimension")
        if "Z" not in dims and radius:
            raise ValueError("source has no Z dimension")
        indexer.extend((slice(None), slice(None)))
        selected_plane = values[tuple(indexer)]
        if projection_axis is not None:
            remaining_axis = projection_axis - sum(
                isinstance(item, int) for item in indexer[:projection_axis]
            )
            selected_plane = cast(
                RasterArray, np.max(selected_plane, axis=remaining_axis)
            )
        return cast(
            RasterPlaneArray, np.ascontiguousarray(selected_plane, dtype=values.dtype)
        )
