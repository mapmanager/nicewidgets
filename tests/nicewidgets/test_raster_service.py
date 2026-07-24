"""Tests for RasterViewService rendering policy."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from nicewidgets.raster_viewer.backend.image_model import (
    BackendImage,
    RasterDisplayStyle,
    RasterGridSpec,
    RowColBounds,
    ViewRequest,
    ViewportSize,
)
from nicewidgets.raster_viewer.backend.pyramid import ImagePyramid
from nicewidgets.raster_viewer.backend.raster_service import RasterViewService


def _service(*, shape: tuple[int, int] = (32, 64), heatmap_max_values: int = 500_000) -> RasterViewService:
    """Build a small RasterViewService for tests."""
    rng = np.random.default_rng(0)
    data = rng.random(shape, dtype=np.float32)
    grid = RasterGridSpec(dx=0.5, dy=0.25, x_unit="row", y_unit="col")
    source = BackendImage(data=data, grid=grid)
    pyramid = ImagePyramid(source)
    return RasterViewService(source=source, pyramid=pyramid, heatmap_max_values=heatmap_max_values)


def _small_fixture_service() -> RasterViewService:
    """Build the 16x32 fixture service used by legacy raster_viewer tests."""
    data = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='', y_unit='')
    source = BackendImage(data=data, grid=grid)
    pyramid = ImagePyramid(source)
    return RasterViewService(source=source, pyramid=pyramid, heatmap_max_values=20)


def _decode_png_rgb(data_uri: str) -> np.ndarray:
    """Decode a PNG data URI into an RGB ndarray ``(rows, cols, 3)``."""
    raw = base64.b64decode(data_uri.split(',', 1)[1])
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert('RGB'))


# ---- normalize_to_uint8 ----


def test_normalize_to_uint8_empty_returns_zeros() -> None:
    """Empty input should return a zero-shaped uint8 array."""
    arr = np.zeros((0, 0), dtype=np.float32)
    out = RasterViewService.normalize_to_uint8(arr)

    assert out.shape == (0, 0)
    assert out.dtype == np.uint8


def test_normalize_to_uint8_handles_nan_array() -> None:
    """An all-NaN input should return zeros without raising."""
    arr = np.full((3, 3), np.nan, dtype=np.float32)
    out = RasterViewService.normalize_to_uint8(arr)

    assert out.dtype == np.uint8
    assert (out == 0).all()


def test_normalize_to_uint8_handles_constant_array() -> None:
    """A constant input where hi <= lo should return zeros."""
    arr = np.full((4, 4), 5.0, dtype=np.float32)
    out = RasterViewService.normalize_to_uint8(arr)

    assert (out == 0).all()


def test_normalize_to_uint8_scales_range_to_uint8() -> None:
    """A linear array should map to 0..255 in uint8."""
    arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = RasterViewService.normalize_to_uint8(arr)

    assert out[0, 0] == 0
    assert out[0, -1] == 255
    assert 110 <= out[0, 1] <= 135


# ---- array_to_png_data_uri ----


def test_array_to_png_data_uri_returns_png_prefix() -> None:
    """Output should start with the PNG data URI prefix."""
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    uri = RasterViewService.array_to_png_data_uri(arr, style=RasterDisplayStyle())

    assert uri.startswith("data:image/png;base64,")


def test_array_to_png_data_uri_handles_empty() -> None:
    """An empty array should still produce a valid 1x1 PNG."""
    arr = np.zeros((0, 0), dtype=np.float32)
    uri = RasterViewService.array_to_png_data_uri(arr, style=RasterDisplayStyle())

    assert uri.startswith("data:image/png;base64,")


def test_array_to_png_data_uri_handles_all_nan() -> None:
    """All-NaN array should produce a valid PNG (zeroed)."""
    arr = np.full((4, 4), np.nan, dtype=np.float32)
    uri = RasterViewService.array_to_png_data_uri(arr, style=RasterDisplayStyle())

    assert uri.startswith("data:image/png;base64,")


def test_array_to_png_data_uri_uses_provided_zmin_zmax() -> None:
    """Providing explicit zmin/zmax should not raise and should produce a PNG."""
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    style = RasterDisplayStyle(colorscale="Viridis", zmin=0.0, zmax=3.0)
    uri = RasterViewService.array_to_png_data_uri(arr, style=style)

    assert uri.startswith("data:image/png;base64,")


# ---- to_png_data_uri (grayscale) ----


def test_to_png_data_uri_returns_png_prefix() -> None:
    """Legacy grayscale PNG path should also return PNG data URI."""
    arr = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    uri = RasterViewService.to_png_data_uri(arr)

    assert uri.startswith("data:image/png;base64,")


# ---- choose_level ----


def test_choose_level_picks_level_zero_for_full_resolution_viewport() -> None:
    """Visible rows ~= viewport width should map to level 0."""
    svc = _service(shape=(64, 64))
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=64, col_min=0, col_max=64),
        viewport=ViewportSize(width_px=64, height_px=64),
    )

    assert svc.choose_level(request) == 0


def test_choose_level_picks_coarser_level_for_small_viewport() -> None:
    """A small viewport relative to large bounds should pick a coarser level."""
    svc = _service(shape=(256, 256))
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=256, col_min=0, col_max=256),
        viewport=ViewportSize(width_px=16, height_px=16),
    )

    level = svc.choose_level(request)
    assert level >= 1


def _kymograph_service() -> RasterViewService:
    """Build a skinny kymograph-shaped service like production data."""
    rng = np.random.default_rng(0)
    data = rng.random((30_000, 24), dtype=np.float32)
    grid = RasterGridSpec(dx=0.001, dy=1.0, x_unit='s', y_unit='ch')
    source = BackendImage(data=data, grid=grid)
    pyramid = ImagePyramid(source)
    return RasterViewService(source=source, pyramid=pyramid)


def test_choose_level_refines_when_spatial_axis_truncated() -> None:
    """Time-driven density must not pick a level that truncates spatial coverage."""
    svc = _kymograph_service()
    request = ViewRequest(
        bounds=RowColBounds(row_min=3625.67, row_max=26132.93, col_min=0.0, col_max=24.0),
        viewport=ViewportSize(width_px=800, height_px=400),
    )

    level = svc.choose_level(request)
    ds = svc.pyramid.get_downsample(level)
    arr = svc.pyramid.get_level(level)

    assert ds <= 4
    assert arr.shape[1] >= int(np.ceil(24.0 / ds))


def test_choose_level_allows_coarser_level_for_partial_spatial_extent() -> None:
    """Zooming the bottom spatial half should not force unnecessary refinement."""
    svc = _kymograph_service()
    request = ViewRequest(
        bounds=RowColBounds(row_min=0.0, row_max=30_000.0, col_min=0.0, col_max=12.0),
        viewport=ViewportSize(width_px=800, height_px=400),
    )

    level = svc.choose_level(request)
    ds = svc.pyramid.get_downsample(level)

    assert ds >= 4


def test_render_spatial_extent_covers_full_col_bounds() -> None:
    """Regression: full spatial extent must tile through col_max after render."""
    svc = _kymograph_service()
    bounds = RowColBounds(row_min=3625.67, row_max=26132.93, col_min=0.0, col_max=24.0)
    request = ViewRequest(bounds=bounds, viewport=ViewportSize(width_px=800, height_px=400))

    response = svc.render(request)

    assert response.mode == 'heatmap_z'
    assert response.z is not None
    ds = float(svc.pyramid.get_downsample(response.level))
    col0 = float(np.floor(min(bounds.col_min, bounds.col_max) / ds) * ds)
    spatial_bins = int(response.z.shape[0])
    covered_col_hi = col0 + spatial_bins * ds
    assert covered_col_hi >= bounds.col_max
    assert spatial_bins > 1


def test_render_square_image_full_bounds_coverage() -> None:
    """Square images use the same choose/render path and cover both axes."""
    rng = np.random.default_rng(2)
    data = rng.random((1024, 1024), dtype=np.float32)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='um', y_unit='um')
    source = BackendImage(data=data, grid=grid)
    pyramid = ImagePyramid(source)
    svc = RasterViewService(source=source, pyramid=pyramid)
    bounds = RowColBounds(row_min=0.0, row_max=1024.0, col_min=0.0, col_max=1024.0)
    request = ViewRequest(bounds=bounds, viewport=ViewportSize(width_px=512, height_px=512))

    response = svc.render(request)

    assert response.z is not None
    ds = float(svc.pyramid.get_downsample(response.level))
    assert int(np.ceil(bounds.row_max / ds)) <= svc.pyramid.get_level(response.level).shape[0]
    assert int(np.ceil(bounds.col_max / ds)) <= svc.pyramid.get_level(response.level).shape[1]


# ---- choose_mode ----


def test_choose_mode_returns_heatmap_for_small_clip() -> None:
    """A small clip should default to heatmap_z."""
    svc = _service(shape=(8, 8))
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=8, col_min=0, col_max=8),
        viewport=ViewportSize(width_px=200, height_px=200),
    )
    clip = np.zeros((8, 8), dtype=np.float32)

    assert svc.choose_mode(request, clip) == "heatmap_z"


def test_choose_mode_returns_png_for_large_clip() -> None:
    """A clip with more values than the threshold should choose image_png."""
    svc = _service(shape=(64, 64), heatmap_max_values=16)
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=64, col_min=0, col_max=64),
        viewport=ViewportSize(width_px=64, height_px=64),
    )
    clip = np.zeros((10, 10), dtype=np.float32)

    assert svc.choose_mode(request, clip) == "image_png"


def test_choose_mode_respects_prefer_mode_override() -> None:
    """A request with prefer_mode should be honored."""
    svc = _service()
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=8, col_min=0, col_max=8),
        viewport=ViewportSize(width_px=100, height_px=100),
        prefer_mode="image_png",
    )
    clip = np.zeros((2, 2), dtype=np.float32)

    assert svc.choose_mode(request, clip) == "image_png"


# ---- render ----


def test_render_returns_heatmap_response_with_z_array() -> None:
    """A small viewport should yield a heatmap_z response."""
    svc = _service(shape=(8, 8))
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=8, col_min=0, col_max=8),
        viewport=ViewportSize(width_px=200, height_px=200),
    )

    response = svc.render(request)

    assert response.mode == "heatmap_z"
    assert response.z is not None
    assert response.z.shape == (8, 8)
    assert response.png_data_uri is None


def test_render_returns_png_response_with_uri() -> None:
    """A request with image_png prefer mode should return a PNG."""
    svc = _service(shape=(8, 8))
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=8, col_min=0, col_max=8),
        viewport=ViewportSize(width_px=200, height_px=200),
        prefer_mode="image_png",
    )

    response = svc.render(request)

    assert response.mode == "image_png"
    assert response.z is None
    assert response.png_data_uri is not None
    assert response.png_data_uri.startswith("data:image/png;base64,")


def test_render_uses_display_style_zmin_zmax_when_provided() -> None:
    """Explicit zmin/zmax in display style should be propagated to the response."""
    svc = _service(shape=(8, 8))
    request = ViewRequest(
        bounds=RowColBounds(row_min=0, row_max=8, col_min=0, col_max=8),
        viewport=ViewportSize(width_px=200, height_px=200),
    )

    style = RasterDisplayStyle(zmin=0.1, zmax=0.9)
    response = svc.render(request, display_style=style)

    assert response.zmin == 0.1
    assert response.zmax == 0.9


def test_render_clips_to_source_shape() -> None:
    """Bounds outside the source shape should be clipped to the source size."""
    svc = _service(shape=(8, 8))
    request = ViewRequest(
        bounds=RowColBounds(row_min=-10, row_max=100, col_min=-5, col_max=100),
        viewport=ViewportSize(width_px=200, height_px=200),
        prefer_mode="image_png",
    )

    response = svc.render(request)

    assert response.bounds.row_min == 0.0
    assert response.bounds.row_max == 8.0
    assert response.bounds.col_min == 0.0
    assert response.bounds.col_max == 8.0


# ---- full_image_png ----


def test_full_image_png_returns_png_for_default_level() -> None:
    """`full_image_png` should always return a PNG response."""
    svc = _service(shape=(16, 16))

    response = svc.full_image_png()

    assert response.mode == "image_png"
    assert response.png_data_uri is not None
    assert response.bounds.row_max == 16.0
    assert response.bounds.col_max == 16.0


def test_full_image_png_respects_explicit_level() -> None:
    """An explicit level should be used to compute downsample / shape."""
    svc = _service(shape=(32, 32))
    response = svc.full_image_png(level=0)

    assert response.level == 0
    assert response.dx == svc.grid.dx
    assert response.dy == svc.grid.dy


def test_full_image_png_uses_display_style() -> None:
    """A custom display style should not raise and should produce a PNG."""
    svc = _service(shape=(16, 16))
    response = svc.full_image_png(display_style=RasterDisplayStyle(colorscale="Viridis"))

    assert response.png_data_uri is not None


# ---- property accessors ----


def test_service_exposes_source_pyramid_grid() -> None:
    """Service should expose its source, pyramid, and grid via properties."""
    svc = _service(shape=(8, 8))

    assert isinstance(svc.source, BackendImage)
    assert isinstance(svc.pyramid, ImagePyramid)
    assert svc.grid.dx > 0


# ---- legacy raster_viewer/tests coverage (ported) ----


def test_choose_level_prefers_coarser_level_for_zoomed_out_view() -> None:
    """Zoomed-out views should prefer a coarser pyramid level."""
    raster_service = _small_fixture_service()
    request = ViewRequest(
        bounds=RowColBounds(row_min=0.0, row_max=8.0, col_min=0.0, col_max=16.0),
        viewport=ViewportSize(width_px=8, height_px=4),
    )
    level = raster_service.choose_level(request)
    # 16x32 fixture builds levels 0 (512 px) and 1 (128 px) only.
    assert level == 1


def test_full_image_png_default_uses_coarse_overview() -> None:
    """Without a pixel budget, the overview uses the conservative coarse level."""
    raster_service = _small_fixture_service()
    response = raster_service.full_image_png()
    assert response.level == raster_service.pyramid.num_levels - 1


def test_full_image_png_max_pixels_selects_finest_fitting_level() -> None:
    """A generous budget selects the finest (full-resolution) level."""
    raster_service = _small_fixture_service()
    response = raster_service.full_image_png(max_pixels=16 * 32)
    assert response.level == 0


def test_full_image_png_max_pixels_steps_to_coarser_level() -> None:
    """A tight budget selects the finest level whose size fits the budget."""
    raster_service = _small_fixture_service()
    # Level 0 = 512 px (too big), level 1 = 128 px (fits first).
    response = raster_service.full_image_png(max_pixels=128)
    assert response.level == 1


def test_full_image_png_max_pixels_too_small_uses_coarsest_level() -> None:
    """When no level fits the budget, the coarsest level is used."""
    raster_service = _small_fixture_service()
    response = raster_service.full_image_png(max_pixels=1)
    assert response.level == raster_service.pyramid.num_levels - 1


def test_full_image_png_explicit_level_overrides_max_pixels() -> None:
    """An explicit ``level`` takes precedence over ``max_pixels``."""
    raster_service = _small_fixture_service()
    response = raster_service.full_image_png(level=1, max_pixels=16 * 32)
    assert response.level == 1


def test_png_greys_matches_plotly_js_direction() -> None:
    """``Greys`` PNG must map low->dark, high->bright to match the Plotly.js heatmap."""
    arr = np.array([[0.0, 255.0]], dtype=np.float32)
    style = RasterDisplayStyle(colorscale='Greys', zmin=0.0, zmax=255.0)
    rgb = _decode_png_rgb(RasterViewService.array_to_png_data_uri(arr, style=style))
    low, high = rgb[0, 0], rgb[0, 1]
    assert int(low.mean()) < int(high.mean())
    np.testing.assert_array_equal(low, (0, 0, 0))
    np.testing.assert_array_equal(high, (255, 255, 255))


def test_png_explicit_inverted_grays_unaffected() -> None:
    """Explicit stop lists are read literally; the reversal fix must not touch them."""
    arr = np.array([[0.0, 255.0]], dtype=np.float32)
    inverted = [[0, 'rgb(255,255,255)'], [1, 'rgb(0,0,0)']]
    style = RasterDisplayStyle(colorscale=inverted, zmin=0.0, zmax=255.0)
    rgb = _decode_png_rgb(RasterViewService.array_to_png_data_uri(arr, style=style))
    np.testing.assert_array_equal(rgb[0, 0], (255, 255, 255))
    np.testing.assert_array_equal(rgb[0, 1], (0, 0, 0))


def test_render_heatmap_uses_display_style_z_window() -> None:
    """Pinned z-range from :class:`RasterDisplayStyle` should appear on heatmap responses."""
    raster_service = _small_fixture_service()
    request = ViewRequest(
        bounds=RowColBounds(row_min=0.0, row_max=2.0, col_min=0.0, col_max=4.0),
        viewport=ViewportSize(width_px=400, height_px=200),
    )
    style = RasterDisplayStyle(zmin=-1.0, zmax=2.0)
    response = raster_service.render(request, display_style=style)
    assert response.mode == 'heatmap_z'
    assert response.zmin == -1.0
    assert response.zmax == 2.0
