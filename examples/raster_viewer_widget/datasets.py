"""Typed dataset models and deterministic synthetic raster examples."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import tifffile

LOGGER = logging.getLogger(__name__)

type RasterArray = npt.NDArray[np.uint16] | npt.NDArray[np.float32]
type RasterPlane = npt.NDArray[np.uint16] | npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RasterHeader:
    """Describe an AcqStore-aligned, channel-independent raster header.

    Attributes:
        shape: Source NumPy plane shape, excluding the channel dimension.
        dims: Dimension names corresponding positionally to ``shape``.
        dtype: Shared NumPy dtype name for every channel.
        num_channels: Number of same-shaped channel arrays.
        physical_units: Sample spacing corresponding positionally to ``dims``.
        physical_units_labels: Unit labels corresponding positionally to ``dims``.
        path: Optional source path.
        date: Optional acquisition date.
        time: Optional acquisition time.
        file_size: Optional human-readable source file size.
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
        """Validate positional dimension metadata.

        Raises:
            ValueError: If dimensions, shape, channel count, or calibration are invalid.
        """
        if len(self.shape) not in (2, 3, 4):
            raise ValueError("header shape must describe a 2D, 3D, or 4D raster")
        if not (
            len(self.shape)
            == len(self.dims)
            == len(self.physical_units)
            == len(self.physical_units_labels)
        ):
            raise ValueError("header positional metadata lengths must match")
        if self.dims[-2:] != ("Y", "X") or any(
            dim not in {"T", "Z"} for dim in self.dims[:-2]
        ):
            raise ValueError("header dims may contain T/Z before ('Y', 'X')")
        if len(set(self.dims)) != len(self.dims) or any(not dim for dim in self.dims):
            raise ValueError("header dims must be unique non-empty names")
        if any(size <= 0 for size in self.shape):
            raise ValueError("header shape values must be positive")
        if self.num_channels <= 0:
            raise ValueError("header num_channels must be positive")
        if any(not np.isfinite(value) or value <= 0 for value in self.physical_units):
            raise ValueError("header physical_units must be finite and positive")

    @property
    def sizes(self) -> dict[str, int]:
        """Return dimension sizes derived from ``dims`` and ``shape``.

        Returns:
            Mapping from each dimension name to its source-array length.
        """
        return dict(zip(self.dims, self.shape, strict=True))

    def to_json(self) -> dict[str, object]:
        """Return a JSON-compatible representation.

        Returns:
            AcqStore-aligned raster header using snake_case keys only.
        """
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
class RasterChannel:
    """Store one scalar channel and its initial display settings.

    Attributes:
        channel_id: Stable identifier within the owning dataset.
        label: Human-readable channel name.
        values: Two- or three-dimensional uint16 or float32 raster.
        default_lut: Initial JavaScript color lookup table name.
    """

    channel_id: str
    label: str
    values: RasterArray
    default_lut: str

    def __post_init__(self) -> None:
        """Validate and normalize the channel array.

        Raises:
            ValueError: If the channel is not a non-empty 2D or 3D array.
            TypeError: If the dtype is not uint16 or float32.
        """
        if self.values.ndim not in (2, 3, 4) or 0 in self.values.shape:
            raise ValueError("channel values must be a non-empty 2D, 3D, or 4D array")
        if self.values.dtype not in (np.dtype(np.uint16), np.dtype(np.float32)):
            raise TypeError("channel values must use uint16 or float32")
        canonical_dtype = np.dtype("<u2") if self.values.dtype.kind == "u" else np.dtype("<f4")
        normalized = np.ascontiguousarray(self.values, dtype=canonical_dtype)
        object.__setattr__(self, "values", normalized)

    @property
    def encoding(self) -> str:
        """Return the raw binary encoding name.

        Returns:
            ``raw-u16-le`` for uint16 or ``raw-f32-le`` for float32.
        """
        return "raw-u16-le" if self.values.dtype.kind == "u" else "raw-f32-le"

    @property
    def served_dtype(self) -> str:
        """Return the browser-facing dtype name.

        Returns:
            ``uint16`` or ``float32``.
        """
        return "uint16" if self.values.dtype.kind == "u" else "float32"

    def binary_bytes(self) -> bytes:
        """Encode the channel as little-endian C-order bytes.

        Returns:
            Raw bytes with no header or compression.
        """
        return self.values.tobytes(order="C")


@dataclass(frozen=True, slots=True)
class RasterDataset:
    """Store one same-shaped collection of raster channels.

    Attributes:
        dataset_id: Stable identifier used by demo routes.
        label: Human-readable dataset name.
        channels: One or more identically shaped scalar channels.
        header: Shared AcqStore-aligned source-array metadata.
    """

    dataset_id: str
    label: str
    channels: tuple[RasterChannel, ...]
    header: RasterHeader

    def __post_init__(self) -> None:
        """Validate identifiers, channel count, uniqueness, and shared shape.

        Raises:
            ValueError: If identifiers are empty, channels are absent, channel IDs
                repeat, or channel shapes differ.
        """
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be empty")
        if not self.channels:
            raise ValueError("a dataset must contain at least one channel")
        channel_ids = [channel.channel_id for channel in self.channels]
        if len(channel_ids) != len(set(channel_ids)):
            raise ValueError("channel IDs must be unique within a dataset")
        expected_shape = self.channels[0].values.shape
        if any(channel.values.shape != expected_shape for channel in self.channels[1:]):
            raise ValueError("all channels in a dataset must have the same shape")
        expected_dtype = self.channels[0].served_dtype
        if any(channel.served_dtype != expected_dtype for channel in self.channels[1:]):
            raise ValueError("all channels in a dataset must have the same dtype")
        if self.header.shape != expected_shape:
            raise ValueError("header shape must match every channel")
        if self.header.dtype != expected_dtype:
            raise ValueError("header dtype must match every channel")
        if self.header.num_channels != len(self.channels):
            raise ValueError("header num_channels must match the channel collection")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the complete shared source-array shape.

        Returns:
            Shared two-dimensional channel shape.
        """
        return self.header.shape

    @property
    def plane_shape(self) -> tuple[int, int]:
        """Return the final source ``(Y, X)`` plane shape.

        Returns:
            Height and width of every displayable channel plane.
        """
        height, width = self.header.shape[-2:]
        return height, width

    def get_plane(
        self,
        channel_id: str,
        t_index: int | None = None,
        z_index: int | None = None,
        plus_minus_z: int = 0,
    ) -> RasterPlane:
        """Return one 2D plane or centered sliding-Z maximum projection.

        Args:
            channel_id: Stable channel identifier.
            t_index: Zero-based T index when the dataset has a T dimension.
            z_index: Zero-based Z center when the dataset has a Z dimension.
            plus_minus_z: Non-negative projection radius around ``z_index``.

        Returns:
            Contiguous 2D plane retaining the source channel dtype.

        Raises:
            KeyError: If ``channel_id`` is unknown.
            ValueError: If plane parameters do not match the dataset dimensions.
            IndexError: If ``z_index`` is outside the Z extent.
        """
        channel = next((item for item in self.channels if item.channel_id == channel_id), None)
        if channel is None:
            raise KeyError(channel_id)
        if plus_minus_z < 0:
            raise ValueError("plus_minus_z must be non-negative")
        selections = {"T": t_index, "Z": z_index}
        indexer: list[int | slice] = []
        projection_axis: int | None = None
        for axis, dim in enumerate(self.header.dims[:-2]):
            selected = selections[dim]
            if selected is None:
                raise ValueError(f"dataset requires {dim.lower()}_index")
            if not 0 <= selected < channel.values.shape[axis]:
                raise IndexError(f"{dim.lower()}_index is outside the dataset extent")
            if dim == "Z" and plus_minus_z:
                indexer.append(
                    slice(
                        max(0, selected - plus_minus_z),
                        min(channel.values.shape[axis], selected + plus_minus_z + 1),
                    )
                )
                projection_axis = axis
            else:
                indexer.append(selected)
        for dim, selected in selections.items():
            if dim not in self.header.dims and selected is not None:
                raise ValueError(f"dataset has no {dim} dimension")
        if "Z" not in self.header.dims and plus_minus_z:
            raise ValueError("dataset has no Z dimension")
        indexer.extend((slice(None), slice(None)))
        plane = channel.values[tuple(indexer)]
        if projection_axis is not None:
            remaining_axis = projection_axis - sum(
                isinstance(item, int) for item in indexer[:projection_axis]
            )
            plane = cast(RasterArray, np.max(plane, axis=remaining_axis))
        return cast(RasterPlane, np.ascontiguousarray(plane, dtype=channel.values.dtype))


class SyntheticDatasetFactory:
    """Create deterministic datasets that expose orientation and rendering errors."""

    def __init__(self, data_directory: Path | None = None) -> None:
        """Initialize the factory with an optional local TIFF directory.

        Args:
            data_directory: Directory containing optional file-backed demo data.
                When omitted, only generated datasets are returned.
        """
        self._data_directory = data_directory

    def create_all(self) -> tuple[RasterDataset, ...]:
        """Create every first-milestone synthetic dataset.

        Returns:
            Deterministic 2D, multichannel, Z-stack, and T/Z datasets.
        """
        generated = (
            self._square_uint16(),
            self._linescan_uint16(),
            self._float32_features(),
            self._three_channel_composite(),
            self._single_channel_z_uint16(),
            self._time_z_uint16(),
        )
        return generated + self._load_rr30a_if_available()

    def _load_rr30a_if_available(self) -> tuple[RasterDataset, ...]:
        """Load optional projected and Z-stack ``rr30a`` datasets.

        Returns:
            Projected and source-Z datasets when both TIFF files exist, otherwise
            an empty tuple when neither file exists.

        Raises:
            FileNotFoundError: If only one required TIFF file exists.
            TypeError: If a TIFF does not contain uint16 samples.
            ValueError: If a TIFF is not shaped ``(70, 1024, 1024)`` or the
                resulting channel projections do not match.
        """
        if self._data_directory is None:
            return ()
        paths = (
            self._data_directory / "rr30a_s0_ch1.tif",
            self._data_directory / "rr30a_s0_ch2.tif",
        )
        existing = tuple(path.exists() for path in paths)
        if not any(existing):
            LOGGER.info("Optional rr30a TIFF dataset not found in %s", self._data_directory)
            return ()
        if not all(existing):
            missing = next(path for path, exists in zip(paths, existing, strict=True) if not exists)
            raise FileNotFoundError(f"rr30a dataset is incomplete; missing {missing}")

        stacks = tuple(self._load_tiff_stack(path) for path in paths)
        projections = tuple(
            np.ascontiguousarray(np.max(stack, axis=0), dtype=np.dtype("<u2"))
            for stack in stacks
        )
        LOGGER.info("Loaded rr30a dataset from %s", self._data_directory)
        projection_dataset = self._dataset(
            "rr30a",
            "rr30a two-channel max projection",
            (
                ("channel_1", projections[0], "green"),
                ("channel_2", projections[1], "magenta"),
            ),
            physical_units=(0.15, 0.15),
            physical_units_labels=("um", "um"),
        )
        zstack_dataset = self._dataset(
            "rr30a_zstack",
            "rr30a two-channel Z stack",
            (
                ("channel_1", stacks[0], "green"),
                ("channel_2", stacks[1], "magenta"),
            ),
            physical_units=(1.0, 0.15, 0.15),
            physical_units_labels=("slice", "um", "um"),
            dims=("Z", "Y", "X"),
        )
        return projection_dataset, zstack_dataset

    def _load_tiff_stack(self, path: Path) -> npt.NDArray[np.uint16]:
        """Load one contiguous uint16 TIFF Z stack.

        Args:
            path: TIFF file expected to contain a ``(70, 1024, 1024)`` stack.

        Returns:
            Contiguous ``(70, 1024, 1024)`` uint16 stack.

        Raises:
            TypeError: If the TIFF dtype is not uint16.
            ValueError: If the TIFF shape is not ``(70, 1024, 1024)``.
        """
        LOGGER.info("Loading TIFF stack %s", path)
        stack = tifffile.imread(path)
        if stack.shape != (70, 1024, 1024):
            raise ValueError(f"expected TIFF shape (70, 1024, 1024), got {stack.shape} for {path}")
        if stack.dtype != np.dtype(np.uint16):
            raise TypeError(f"expected uint16 TIFF data, got {stack.dtype} for {path}")
        return np.ascontiguousarray(stack, dtype=np.dtype("<u2"))

    def _square_uint16(self) -> RasterDataset:
        """Create a square uint16 orientation target.

        Returns:
            Two-channel square dataset with asymmetric gradients and landmarks.
        """
        height = width = 256
        y, x = np.mgrid[0:height, 0:width]
        first = np.clip(x * 180 + y * 35, 0, 65535).astype(np.uint16)
        first[18:62, 24:90] = 65535
        second = np.clip(y * 190 + (width - x) * 28, 0, 65535).astype(np.uint16)
        second[168:232, 154:224] = 60000
        return self._dataset(
            "square_uint16",
            "Square uint16 orientation",
            (("horizontal", first, "green"), ("vertical", second, "magenta")),
            physical_units=(0.25, 0.25),
            physical_units_labels=("µm", "µm"),
        )

    def _linescan_uint16(self) -> RasterDataset:
        """Create a non-square multichannel linescan.

        Returns:
            Two-channel uint16 dataset with moving spatial features.
        """
        height, width = 30_000, 200
        y = np.arange(height, dtype=np.float32)[:, np.newaxis]
        x = np.arange(width, dtype=np.float32)[np.newaxis, :]
        center_a = 52 + 18 * np.sin(y / 42)
        center_b = 132 + 12 * np.cos(y / 55)
        first = (52000 * np.exp(-((x - center_a) ** 2) / 150) + 1200).astype(np.uint16)
        second = (47000 * np.exp(-((x - center_b) ** 2) / 210) + 900).astype(np.uint16)
        return self._dataset(
            "linescan_uint16",
            "Two-channel uint16 linescan",
            (("calcium", first, "green"), ("vessel", second, "magenta")),
            physical_units=(0.001, 0.2),
            physical_units_labels=("seconds", "um"),
        )

    def _float32_features(self) -> RasterDataset:
        """Create float32 data containing negative and non-finite samples.

        Returns:
            One-channel float32 dataset.
        """
        height, width = 180, 300
        y, x = np.mgrid[0:height, 0:width]
        values = (np.sin(x / 19) * 2.5 + np.cos(y / 14) - 0.006 * y).astype(np.float32)
        values[12, 18] = np.nan
        values[13, 18] = np.inf
        return self._dataset(
            "float32_features",
            "Float32 signed features",
            (("response", values, "viridis"),),
            physical_units=(0.01, 1.0),
            physical_units_labels=("s", "px"),
        )

    def _three_channel_composite(self) -> RasterDataset:
        """Create three overlapping channels for composite validation.

        Returns:
            Three-channel uint16 dataset with offset Gaussian spots.
        """
        height, width = 240, 320
        y, x = np.mgrid[0:height, 0:width]

        def spot(cx: float, cy: float, sigma: float) -> npt.NDArray[np.uint16]:
            """Create one Gaussian uint16 spot.

            Args:
                cx: Horizontal spot center in pixels.
                cy: Vertical spot center in pixels.
                sigma: Gaussian standard deviation in pixels.

            Returns:
                Two-dimensional uint16 spot image.
            """
            result = 62000 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
            return cast(
                npt.NDArray[np.uint16],
                np.clip(result + 500, 0, 65535).astype(np.uint16),
            )

        return self._dataset(
            "three_channel_composite",
            "Three-channel composite",
            (
                ("red_feature", spot(118, 110, 42), "red"),
                ("green_feature", spot(176, 92, 38), "green"),
                ("blue_feature", spot(158, 154, 46), "cyan"),
            ),
            physical_units=(0.5, 0.5),
            physical_units_labels=("µm", "µm"),
        )

    def _single_channel_z_uint16(self) -> RasterDataset:
        """Create a calibrated single-channel Z stack for viewer layout tests.

        Returns:
            One-channel ``(Z, Y, X)`` uint16 dataset whose moving features make
            plane changes and orientation easy to recognize.
        """
        z_size, height, width = 20, 512, 512
        y, x = np.ogrid[:height, :width]
        values = np.empty((z_size, height, width), dtype=np.uint16)
        for z_index in range(z_size):
            first_x = 110 + z_index * 12
            first_y = 150 + z_index * 7
            second_x = 390 - z_index * 8
            second_y = 360 - z_index * 9
            first = 54_000 * np.exp(
                -((x - first_x) ** 2 + (y - first_y) ** 2) / (2 * 34**2)
            )
            second = 37_000 * np.exp(
                -((x - second_x) ** 2 + (y - second_y) ** 2) / (2 * 48**2)
            )
            background = 350 + x * 8 + y * 3
            values[z_index] = np.clip(background + first + second, 0, 65535)

        return self._dataset(
            "single_channel_z_uint16",
            "Single-channel uint16 Z stack",
            (("channel_0", values, "viridis"),),
            physical_units=(0.75, 0.4, 0.2),
            physical_units_labels=("um", "um", "um"),
            dims=("Z", "Y", "X"),
        )

    def _time_z_uint16(self) -> RasterDataset:
        """Create a compact two-channel T/Z stack for multidimensional smoke tests."""
        t_size, z_size, height, width = 4, 6, 96, 128
        y, x = np.mgrid[0:height, 0:width]
        channels: list[npt.NDArray[np.uint16]] = []
        for offset in (0.0, 24.0):
            values = np.empty((t_size, z_size, height, width), dtype=np.uint16)
            for t_index in range(t_size):
                for z_index in range(z_size):
                    cx = 25 + offset + t_index * 8
                    cy = 28 + z_index * 7
                    image = 60000 * np.exp(
                        -((x - cx) ** 2 + (y - cy) ** 2) / (2 * 11**2)
                    )
                    values[t_index, z_index] = np.clip(image + 400, 0, 65535)
            channels.append(values)
        return self._dataset(
            "time_z_uint16",
            "Two-channel T/Z stack",
            (("channel_0", channels[0], "green"), ("channel_1", channels[1], "magenta")),
            physical_units=(0.5, 1.0, 0.25, 0.25),
            physical_units_labels=("s", "slice", "um", "um"),
            dims=("T", "Z", "Y", "X"),
        )

    def _dataset(
        self,
        dataset_id: str,
        label: str,
        channels: tuple[tuple[str, RasterArray, str], ...],
        physical_units: tuple[float, ...],
        physical_units_labels: tuple[str, ...],
        dims: tuple[str, ...] = ("Y", "X"),
    ) -> RasterDataset:
        """Build a dataset from compact channel definitions.

        Args:
            dataset_id: Stable dataset identifier.
            label: Human-readable dataset name.
            channels: Channel ID, array, and default LUT tuples.
            physical_units: Source ``(Y, X)`` sample spacing.
            physical_units_labels: Source ``(Y, X)`` unit labels.
            dims: Source dimension names, excluding channels.

        Returns:
            Validated raster dataset.
        """
        raster_channels = tuple(
            RasterChannel(channel_id, channel_id.replace("_", " ").title(), values, lut)
            for channel_id, values, lut in channels
        )
        dtype = raster_channels[0].served_dtype
        header = RasterHeader(
            shape=cast(tuple[int, ...], raster_channels[0].values.shape),
            dims=dims,
            dtype=dtype,
            num_channels=len(raster_channels),
            physical_units=physical_units,
            physical_units_labels=physical_units_labels,
        )
        return RasterDataset(dataset_id, label, raster_channels, header)
