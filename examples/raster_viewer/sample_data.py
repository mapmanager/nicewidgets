"""Synthetic sample data for the raster viewer demos.

Pure data module: no NiceGUI imports. Each demo dataset provides one or more
2D channels (``float32``, intensity range ``0..255`` so integer contrast
sliders are meaningful) plus a :class:`RasterGridSpec` describing physical
pixel spacing and axis units.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec

INTENSITY_MAX = 255.0

PlaneFactory = Callable[[], np.ndarray]


def _to_uint8_range(data: np.ndarray) -> np.ndarray:
    """Rescale an array to ``float32`` in ``0..255``."""
    lo = float(data.min())
    hi = float(data.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    scaled = (data - lo) / (hi - lo) * INTENSITY_MAX
    return scaled.astype(np.float32)


@dataclass(frozen=True)
class DemoDataset:
    """One selectable demo raster with per-channel factories and grid metadata.

    Args:
        name: Display name shown in the dataset select.
        channel_factories: One factory per channel; each returns a 2D array.
        grid: Physical spacing (``dx``/``dy``) and axis units for all channels.
    """

    name: str
    channel_factories: tuple[PlaneFactory, ...]
    grid: RasterGridSpec

    @property
    def channels(self) -> list[int]:
        """Return channel indices for this dataset."""
        return list(range(len(self.channel_factories)))


class SampleDataCatalog:
    """Provide lazily generated multi-channel demo arrays with grids."""

    def __init__(self) -> None:
        self._datasets: list[DemoDataset] = [
            DemoDataset(
                name='Diagonal bands',
                channel_factories=(
                    self._make_diagonal_bands,
                    self._make_diagonal_bands_inverted,
                ),
                grid=RasterGridSpec(dx=1.0, dy=1.0, x_unit='Pixels', y_unit='Pixels'),
            ),
            DemoDataset(
                name='Gaussian blobs',
                channel_factories=(
                    self._make_gaussian_blobs,
                    self._make_gaussian_ring,
                ),
                grid=RasterGridSpec(dx=0.01, dy=0.02, x_unit='time (s)', y_unit='um'),
            ),
            DemoDataset(
                name='Noisy stripes',
                channel_factories=(
                    self._make_noisy_stripes,
                    self._make_noisy_checker,
                ),
                grid=RasterGridSpec(dx=0.5, dy=1.0, x_unit='index', y_unit='Pixels'),
            ),
        ]
        self._by_name: dict[str, DemoDataset] = {d.name: d for d in self._datasets}

    @property
    def names(self) -> list[str]:
        """Return demo dataset names."""
        return [d.name for d in self._datasets]

    def dataset(self, name: str) -> DemoDataset:
        """Return the dataset definition for ``name``."""
        return self._by_name[name]

    def channels(self, name: str) -> list[int]:
        """Return channel indices for the named dataset."""
        return self.dataset(name).channels

    def grid(self, name: str) -> RasterGridSpec:
        """Return the physical grid for the named dataset."""
        return self.dataset(name).grid

    def get_plane(self, name: str, channel: int) -> np.ndarray:
        """Generate one 2D channel plane (``float32``, ``0..255``).

        Args:
            name: Dataset name from :attr:`names`.
            channel: Channel index from :meth:`channels`.

        Returns:
            2D ``float32`` array scaled to ``0..255``.
        """
        dataset = self.dataset(name)
        factory = dataset.channel_factories[channel]
        return _to_uint8_range(factory())

    @staticmethod
    def _make_diagonal_bands() -> np.ndarray:
        rows, cols = 5000, 1024
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        return 0.5 + 0.5 * np.sin((x * 0.035) + (y * 0.11))

    @staticmethod
    def _make_diagonal_bands_inverted() -> np.ndarray:
        return -SampleDataCatalog._make_diagonal_bands()

    @staticmethod
    def _make_gaussian_blobs() -> np.ndarray:
        rows, cols = 4096, 768
        y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
        x = np.linspace(-3.0, 3.0, cols, dtype=np.float32)[None, :]
        blob1 = np.exp(-(((x + 1.2) ** 2) / 0.12 + ((y + 0.25) ** 2) / 0.08))
        blob2 = 0.8 * np.exp(-(((x - 0.8) ** 2) / 0.45 + ((y - 0.1) ** 2) / 0.03))
        blob3 = 0.6 * np.exp(-(((x - 1.8) ** 2) / 0.20 + ((y + 0.45) ** 2) / 0.10))
        return blob1 + blob2 + blob3

    @staticmethod
    def _make_gaussian_ring() -> np.ndarray:
        rows, cols = 4096, 768
        y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
        x = np.linspace(-3.0, 3.0, cols, dtype=np.float32)[None, :]
        radius = np.sqrt((x / 3.0) ** 2 + y**2)
        return np.exp(-((radius - 0.55) ** 2) / 0.02)

    @staticmethod
    def _make_noisy_stripes() -> np.ndarray:
        rows, cols = 12000, 512
        rng = np.random.default_rng(7)
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        base = 0.45 + 0.25 * np.sin(x * 0.012)
        stripes = 0.20 * np.sin((x * 0.06) + (y * 0.20))
        noise = rng.normal(loc=0.0, scale=0.04, size=(rows, cols)).astype(np.float32)
        return base + stripes + noise

    @staticmethod
    def _make_noisy_checker() -> np.ndarray:
        rows, cols = 12000, 512
        rng = np.random.default_rng(11)
        y = np.arange(rows, dtype=np.float32)[:, None]
        x = np.arange(cols, dtype=np.float32)[None, :]
        checker = np.sign(np.sin(x * 0.05)) * np.sign(np.sin(y * 0.01))
        noise = rng.normal(loc=0.0, scale=0.15, size=(rows, cols)).astype(np.float32)
        return checker + noise
