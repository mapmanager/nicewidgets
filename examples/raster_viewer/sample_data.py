"""Sample data for the raster viewer demos.

Pure data module: no NiceGUI imports. Synthetic datasets provide 2D channels
(``float32``, intensity range ``0..255`` so integer contrast sliders are
meaningful). The ``rr30a`` dataset loads real TIFF stacks from ``data/`` and
exposes a max-intensity projection per channel in native ``uint16`` range.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import tifffile

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec

INTENSITY_MAX = 255.0

_DATA_DIR = Path(__file__).resolve().parent / 'data'

PlaneFactory = Callable[[], np.ndarray]


def _to_uint8_range(data: np.ndarray) -> np.ndarray:
    """Rescale an array to ``float32`` in ``0..255``."""
    lo = float(data.min())
    hi = float(data.max())
    if hi <= lo:
        return np.zeros_like(data, dtype=np.float32)
    scaled = (data - lo) / (hi - lo) * INTENSITY_MAX
    return scaled.astype(np.float32)


@lru_cache(maxsize=4)
def _load_tiff_mip(path: str) -> np.ndarray:
    """Load a TIFF volume and return a max-intensity projection on axis 0.

    Args:
        path: Filesystem path string to a 3D TIFF ``(Z, Y, X)`` (or any stack
            whose first axis should be collapsed).

    Returns:
        2D array in the file's native dtype (``uint16`` for ``rr30a``).
    """
    volume = np.asarray(tifffile.imread(path))
    if volume.ndim == 2:
        return volume
    if volume.ndim != 3:
        raise ValueError(f'expected 2D or 3D TIFF at {path}, got shape={volume.shape}')
    return np.max(volume, axis=0)


@dataclass(frozen=True)
class DemoDataset:
    """One selectable demo raster with per-channel factories and grid metadata.

    Args:
        name: Display name shown in the dataset select.
        channel_factories: One factory per channel; each returns a 2D array.
        grid: Physical spacing (``dx``/``dy``) and axis units for all channels.
        rescale_to_uint8: When ``True``, :meth:`SampleDataCatalog.get_plane`
            rescales to ``0..255`` float32. When ``False``, planes are returned
            as ``float32`` copies of the native values (no min/max stretch).
    """

    name: str
    channel_factories: tuple[PlaneFactory, ...]
    grid: RasterGridSpec
    rescale_to_uint8: bool = True

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
                    self._make_gaussian_blobs_ch1,
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
            DemoDataset(
                name='rr30a',
                channel_factories=(
                    self._make_rr30a_ch1,
                    self._make_rr30a_ch2,
                ),
                grid=RasterGridSpec(dx=0.15, dy=0.15, x_unit='um', y_unit='um'),
                rescale_to_uint8=False,
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
        """Generate one 2D channel plane.

        Synthetic datasets return ``float32`` in ``0..255``. ``rr30a`` returns
        ``float32`` values in the native TIFF intensity range (no stretch).

        Args:
            name: Dataset name from :attr:`names`.
            channel: Channel index from :meth:`channels`.

        Returns:
            2D ``float32`` array.
        """
        dataset = self.dataset(name)
        factory = dataset.channel_factories[channel]
        plane = factory()
        if dataset.rescale_to_uint8:
            return _to_uint8_range(plane)
        return np.asarray(plane, dtype=np.float32)

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
        """Channel 0: sparse bright blobs on zero background (left/center)."""
        rows, cols = 4096, 768
        y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
        x = np.linspace(-3.0, 3.0, cols, dtype=np.float32)[None, :]
        blob1 = np.exp(-(((x + 1.2) ** 2) / 0.12 + ((y + 0.25) ** 2) / 0.08))
        blob2 = 0.85 * np.exp(-(((x - 0.2) ** 2) / 0.10 + ((y - 0.35) ** 2) / 0.06))
        blob3 = 0.7 * np.exp(-(((x + 2.0) ** 2) / 0.08 + ((y - 0.55) ** 2) / 0.05))
        return blob1 + blob2 + blob3

    @staticmethod
    def _make_gaussian_blobs_ch1() -> np.ndarray:
        """Channel 1: sparse bright blobs on zero background (right/offset).

        Spatially offset from channel 0 so composite Red/Green overlap is easy
        to read; background stays near zero (not an inverted fill).
        """
        rows, cols = 4096, 768
        y = np.linspace(-1.0, 1.0, rows, dtype=np.float32)[:, None]
        x = np.linspace(-3.0, 3.0, cols, dtype=np.float32)[None, :]
        blob1 = np.exp(-(((x - 1.1) ** 2) / 0.11 + ((y - 0.20) ** 2) / 0.07))
        blob2 = 0.8 * np.exp(-(((x - 2.0) ** 2) / 0.09 + ((y + 0.40) ** 2) / 0.05))
        blob3 = 0.65 * np.exp(-(((x + 0.3) ** 2) / 0.07 + ((y + 0.55) ** 2) / 0.04))
        return blob1 + blob2 + blob3

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

    @staticmethod
    def _make_rr30a_ch1() -> np.ndarray:
        """Max-intensity projection of ``data/rr30a_s0_ch1.tif`` (native dtype)."""
        return _load_tiff_mip(str(_DATA_DIR / 'rr30a_s0_ch1.tif'))

    @staticmethod
    def _make_rr30a_ch2() -> np.ndarray:
        """Max-intensity projection of ``data/rr30a_s0_ch2.tif`` (native dtype)."""
        return _load_tiff_mip(str(_DATA_DIR / 'rr30a_s0_ch2.tif'))
