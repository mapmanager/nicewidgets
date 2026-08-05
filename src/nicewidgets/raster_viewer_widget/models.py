"""Typed source, plane, channel, and header models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

type RasterPlaneArray = npt.NDArray[np.uint16] | npt.NDArray[np.float32]
RASTER_DESCRIPTOR_SCHEMA_VERSION = "2.0"


@dataclass(frozen=True, slots=True)
class RasterPlaneRequest:
    """Identify one requested channel plane or sliding projection.

    Attributes:
        channel_id: Stable channel identifier.
        t_index: Optional zero-based T index.
        z_index: Optional zero-based Z index.
        plus_minus_z: Non-negative sliding-Z projection radius.
    """

    channel_id: str
    t_index: int | None = None
    z_index: int | None = None
    plus_minus_z: int = 0

    def __post_init__(self) -> None:
        """Validate identity, optional indices, and projection radius."""
        if not self.channel_id.strip():
            raise ValueError("channel_id must not be empty")
        if self.t_index is not None and self.t_index < 0:
            raise ValueError("t_index must be non-negative")
        if self.z_index is not None and self.z_index < 0:
            raise ValueError("z_index must be non-negative")
        if self.plus_minus_z < 0:
            raise ValueError("plus_minus_z must be non-negative")


@dataclass(frozen=True, slots=True)
class RasterChannelDisplay:
    """Describe initial presentation state for one scalar channel.

    Attributes:
        lut: JavaScript color lookup-table identifier.
        value_min: Optional explicit lower contrast bound.
        value_max: Optional explicit upper contrast bound.
        visible: Whether the channel initially contributes to rendering.
    """

    lut: str = "gray"
    value_min: float | None = None
    value_max: float | None = None
    visible: bool = True

    def __post_init__(self) -> None:
        """Validate an optional explicit contrast range."""
        if (self.value_min is None) != (self.value_max is None):
            raise ValueError("value_min and value_max must be supplied together")
        if self.value_min is not None and self.value_max is not None:
            if not np.isfinite(self.value_min) or not np.isfinite(self.value_max):
                raise ValueError("channel display limits must be finite")
            if self.value_min >= self.value_max:
                raise ValueError("value_min must be less than value_max")

    def to_json(self) -> dict[str, object]:
        """Return canonical browser presentation state."""
        return {
            "lut": self.lut,
            "value_min": self.value_min,
            "value_max": self.value_max,
            "visible": self.visible,
        }


@dataclass(frozen=True, slots=True)
class RasterChannelDescriptor:
    """Describe one scalar channel and its initial LUT.

    Attributes:
        channel_id: Stable identifier within the source.
        label: Human-readable channel label.
        display: Initial browser presentation state.
    """

    channel_id: str
    label: str
    display: RasterChannelDisplay = field(default_factory=RasterChannelDisplay)


@dataclass(frozen=True, slots=True)
class RasterHeader:
    """Describe channel-independent source dimensions.

    The header excludes a channel dimension even when an input NumPy array has
    one. ``shape`` and ``dims`` describe each logical scalar channel.

    Attributes:
        shape: Logical per-channel array shape.
        dims: Dimension labels corresponding to ``shape``.
        dtype: Shared ``uint16`` or ``float32`` dtype name.
        num_channels: Number of logical scalar channels.
        physical_units: Sample spacing corresponding to ``dims``.
        physical_units_labels: Unit labels corresponding to ``dims``.
        path: Optional source path.
        date: Optional acquisition date.
        time: Optional acquisition time.
        file_size: Optional human-readable file size.
    """

    shape: tuple[int, ...]
    dims: tuple[str, ...]
    dtype: str
    num_channels: int
    physical_units: tuple[float, ...]
    physical_units_labels: tuple[str, ...]
    path: str = ""
    date: str = ""
    time: str = ""
    file_size: str = ""

    def __post_init__(self) -> None:
        """Validate the canonical per-channel header."""
        if self.dims[-2:] != ("Y", "X"):
            raise ValueError("dims must end with ('Y', 'X')")
        leading = self.dims[:-2]
        if len(set(self.dims)) != len(self.dims) or any(
            dim not in {"T", "Z"} for dim in leading
        ):
            raise ValueError("dims may contain unique T and Z axes before Y/X")
        if not (
            len(self.shape)
            == len(self.dims)
            == len(self.physical_units)
            == len(self.physical_units_labels)
        ):
            raise ValueError("header positional metadata lengths must match")
        if any(size <= 0 for size in self.shape):
            raise ValueError("shape values must be positive")
        if self.dtype not in {"uint16", "float32"}:
            raise ValueError("dtype must be uint16 or float32")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if any(not np.isfinite(value) or value <= 0 for value in self.physical_units):
            raise ValueError("physical_units must be finite and positive")

    @property
    def sizes(self) -> dict[str, int]:
        """Return a mapping from dimension label to length."""
        return dict(zip(self.dims, self.shape, strict=True))

    def to_json(self) -> dict[str, object]:
        """Return canonical snake_case JSON metadata."""
        return {
            "path": self.path,
            "shape": list(self.shape),
            "dims": list(self.dims),
            "sizes": self.sizes,
            "dtype": self.dtype,
            "num_channels": self.num_channels,
            "physical_units": list(self.physical_units),
            "physical_units_labels": list(self.physical_units_labels),
            "date": self.date,
            "time": self.time,
            "file_size": self.file_size,
        }


@dataclass(frozen=True, slots=True)
class RasterDescriptor:
    """Describe a complete browser-loadable raster source.

    Attributes:
        source_id: Stable source identifier.
        label: Human-readable source label.
        header: Shared per-channel header.
        channels: Ordered scalar channel descriptors.
        rois: Initial committed ROI envelopes.
        schema_version: Exact browser descriptor contract version. Consumers
            must reject versions they do not understand instead of guessing.
    """

    source_id: str
    label: str
    header: RasterHeader
    channels: tuple[RasterChannelDescriptor, ...]
    rois: tuple[dict[str, object], ...] = field(default_factory=tuple)
    schema_version: str = RASTER_DESCRIPTOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate source and channel identifiers."""
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.schema_version != RASTER_DESCRIPTOR_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {RASTER_DESCRIPTOR_SCHEMA_VERSION!r}"
            )
        if len(self.channels) != self.header.num_channels:
            raise ValueError("channel count must match header num_channels")
        identifiers = [channel.channel_id for channel in self.channels]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("channel identifiers must be unique")
