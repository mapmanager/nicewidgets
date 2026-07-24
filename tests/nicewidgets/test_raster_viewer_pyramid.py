"""Tests for the raster viewer image pyramid."""

from __future__ import annotations

import numpy as np

from nicewidgets.raster_viewer.backend.image_model import BackendImage, RasterGridSpec, RowColBounds
from nicewidgets.raster_viewer.backend.pyramid import MIN_PYRAMID_AXIS, ImagePyramid

_DEFAULT_GRID = RasterGridSpec(dx=1.0, dy=1.0, x_unit='', y_unit='')


def _image_pyramid(*, rows: int = 16, cols: int = 32) -> ImagePyramid:
    """Return a pyramid built from a small deterministic array."""
    data = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    return ImagePyramid(BackendImage(data, grid=_DEFAULT_GRID))


def test_pyramid_builds_expected_first_levels() -> None:
    """Pyramid should build power-of-two downsample levels until the short axis cap."""
    image_pyramid = _image_pyramid()
    info = image_pyramid.level_info()
    assert info[0].downsample == 1
    assert info[0].shape == (16, 32)
    assert info[1].downsample == 2
    assert info[1].shape == (8, 16)
    assert len(info) == 2


def test_pyramid_stops_before_short_axis_collapses_below_min() -> None:
    """Skinny kymograph arrays should not build single-column pyramid levels."""
    rng = np.random.default_rng(0)
    data = rng.random((30_000, 24), dtype=np.float32)
    grid = RasterGridSpec(dx=0.001, dy=1.0, x_unit='s', y_unit='ch')
    source = BackendImage(data=data, grid=grid)
    pyramid = ImagePyramid(source)

    info = pyramid.level_info()
    coarsest = info[-1]

    assert coarsest.downsample == 4
    assert coarsest.shape == (7_500, 6)
    assert min(coarsest.shape) >= MIN_PYRAMID_AXIS


def test_pyramid_square_image_coarsest_level_respects_min_axis() -> None:
    """Square arrays use the same short-axis cap without a separate code path."""
    rng = np.random.default_rng(1)
    data = rng.random((1024, 1024), dtype=np.float32)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='um', y_unit='um')
    source = BackendImage(data=data, grid=grid)
    pyramid = ImagePyramid(source)

    coarsest = pyramid.level_info()[-1]

    assert coarsest.downsample == 128
    assert coarsest.shape == (8, 8)
    assert min(coarsest.shape) >= MIN_PYRAMID_AXIS


def test_downsample2_averages_2x2_blocks() -> None:
    """Box downsample should average each 2x2 block."""
    data = np.array([[0, 2], [4, 6]], dtype=np.float32)
    out = ImagePyramid._downsample2(data)
    np.testing.assert_allclose(out, np.array([[3.0]], dtype=np.float32))


def test_clip_from_level_uses_source_coordinates() -> None:
    """Level clips should map full-resolution row/col bounds through the downsample."""
    image_pyramid = _image_pyramid()
    bounds = RowColBounds(row_min=4.0, row_max=12.0, col_min=8.0, col_max=24.0)
    clip = image_pyramid.clip_from_level(level=1, bounds=bounds)
    expected = image_pyramid.get_level(1)[2:6, 4:12]
    np.testing.assert_array_equal(clip, expected)
