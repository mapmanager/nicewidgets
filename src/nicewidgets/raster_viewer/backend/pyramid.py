"""Multiresolution pyramid for 2D numpy arrays."""

from __future__ import annotations

import numpy as np

from nicewidgets.raster_viewer.backend.image_model import BackendImage, PyramidLevelInfo, RowColBounds

# Do not build pyramid levels whose shorter side would drop below this size.
# Symmetric 2x downsampling uses one global ``ds`` per level; for skinny arrays
# (e.g. kymographs with few spatial columns) overly coarse levels collapse the
# short axis to one bin and cannot tile the full spatial extent.
MIN_PYRAMID_AXIS: int = 6


class ImagePyramid:
    """Precompute and store downsampled pyramid levels.

    This class accepts a :class:`BackendImage` and creates progressively
    downsampled levels using a fixed 2x box downsample.
    """

    def __init__(self, source: BackendImage, max_levels: int | None = None) -> None:
        """Initialize and build the pyramid.

        Args:
            source: Full-resolution backend image.
            max_levels: Optional maximum number of pyramid levels, including level 0.
        """
        self._source = source
        self._levels: list[np.ndarray] = [source.data]
        self._downsamples: list[int] = [1]
        self._build(max_levels=max_levels)

    def _build(self, max_levels: int | None = None) -> None:
        """Build pyramid levels until exhaustion or ``max_levels``."""
        current = self._source.data
        current_level = 0
        while True:
            if max_levels is not None and current_level + 1 >= max_levels:
                break
            height, width = current.shape
            if height < 2 or width < 2:
                break
            next_level = self._downsample2(current)
            if min(next_level.shape) < MIN_PYRAMID_AXIS:
                break
            self._levels.append(next_level)
            self._downsamples.append(self._downsamples[-1] * 2)
            current = next_level
            current_level += 1

    @staticmethod
    def _downsample2(data: np.ndarray) -> np.ndarray:
        """Downsample a 2D array by a factor of 2 using box averaging.

        Args:
            data: Source 2D array.

        Returns:
            Downsampled float32 array.
        """
        height, width = data.shape
        out_height = height // 2
        out_width = width // 2
        trimmed = data[: out_height * 2, : out_width * 2].astype(np.float32)
        return (
            trimmed[0::2, 0::2]
            + trimmed[1::2, 0::2]
            + trimmed[0::2, 1::2]
            + trimmed[1::2, 1::2]
        ) / 4.0

    @property
    def num_levels(self) -> int:
        """Return the number of pyramid levels."""
        return len(self._levels)

    def level_info(self) -> list[PyramidLevelInfo]:
        """Return metadata for all pyramid levels."""
        return [
            PyramidLevelInfo(level=i, downsample=ds, shape=arr.shape)
            for i, (ds, arr) in enumerate(zip(self._downsamples, self._levels, strict=True))
        ]

    def get_level(self, level: int) -> np.ndarray:
        """Return a pyramid level array.

        Args:
            level: Zero-based pyramid level index.

        Returns:
            The requested level array.
        """
        return self._levels[level]

    def get_downsample(self, level: int) -> int:
        """Return the downsample factor for a pyramid level."""
        return self._downsamples[level]

    def clip_from_level(self, level: int, bounds: RowColBounds) -> np.ndarray:
        """Clip a region from a pyramid level using full-resolution row/column bounds.

        Args:
            level: Pyramid level to sample.
            bounds: Bounds in full-resolution row/column coordinates.

        Returns:
            The clipped 2D array ``(rows, cols)`` from the selected level.
        """
        ds = self.get_downsample(level)
        arr = self.get_level(level)

        r0 = max(0, int(np.floor(min(bounds.row_min, bounds.row_max) / ds)))
        r1 = min(arr.shape[0], int(np.ceil(max(bounds.row_min, bounds.row_max) / ds)))
        c0 = max(0, int(np.floor(min(bounds.col_min, bounds.col_max) / ds)))
        c1 = min(arr.shape[1], int(np.ceil(max(bounds.col_min, bounds.col_max) / ds)))
        return arr[r0:r1, c0:c1]
