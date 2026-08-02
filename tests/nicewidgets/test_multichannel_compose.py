"""Tests for multi-channel RGB composition (Phase 1)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec, RowColBounds
from nicewidgets.raster_viewer.multichannel import (
    MAX_COMPOSITE_CHANNELS,
    ChannelDisplayStyle,
    ChannelPlane,
    CompositeChannelLimitError,
    build_image_rgb_response,
    compose_rgb_png_data_uri,
    compose_rgb_uint8,
    compose_rgb_uint8_for_bounds,
    default_tint_for_channel,
    rgb_rows_cols_to_plotly_image_z,
    select_composite_planes,
    validate_same_shape,
)


def _plane(
    channel_id: int,
    data: np.ndarray,
    *,
    visible: bool = True,
    zmin: float | None = None,
    zmax: float | None = None,
    tint_rgb: tuple[float, float, float] | None = None,
    colorscale: str | list[list[float | str]] = 'Greys',
) -> ChannelPlane:
    return ChannelPlane(
        channel_id=channel_id,
        data=data,
        style=ChannelDisplayStyle(
            visible=visible,
            zmin=zmin,
            zmax=zmax,
            tint_rgb=tint_rgb,
            colorscale=colorscale,
        ),
    )


def test_validate_same_shape_accepts_matching_planes() -> None:
    a = _plane(0, np.zeros((4, 5), dtype=np.float32))
    b = _plane(1, np.ones((4, 5), dtype=np.float32))
    assert validate_same_shape([a, b]) == (4, 5)


def test_validate_same_shape_rejects_mismatch() -> None:
    a = _plane(0, np.zeros((4, 5), dtype=np.float32))
    b = _plane(1, np.ones((4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match='same shape'):
        validate_same_shape([a, b])


def test_select_composite_planes_skips_hidden_and_enforces_limit() -> None:
    planes = [
        _plane(0, np.zeros((2, 2), dtype=np.float32)),
        _plane(1, np.zeros((2, 2), dtype=np.float32), visible=False),
        _plane(2, np.zeros((2, 2), dtype=np.float32)),
    ]
    selected = select_composite_planes(planes)
    assert [p.channel_id for p in selected] == [0, 2]

    too_many = [
        _plane(i, np.zeros((2, 2), dtype=np.float32))
        for i in range(MAX_COMPOSITE_CHANNELS + 1)
    ]
    with pytest.raises(CompositeChannelLimitError, match='at most'):
        select_composite_planes(too_many)


def test_compose_rgb_maps_two_channels_to_red_and_green() -> None:
    """Channel 0 at full intensity → red; channel 1 → green; no blue."""
    red = np.full((3, 3), 100.0, dtype=np.float32)
    green = np.full((3, 3), 100.0, dtype=np.float32)
    rgb = compose_rgb_uint8(
        [
            _plane(0, red, zmin=0.0, zmax=100.0),
            _plane(1, green, zmin=0.0, zmax=100.0),
        ]
    )
    assert rgb.shape == (3, 3, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[..., 0] == 255)
    assert np.all(rgb[..., 1] == 255)
    assert np.all(rgb[..., 2] == 0)


def test_compose_rgb_respects_per_channel_window() -> None:
    """Values below zmin contribute nothing after normalization."""
    data = np.array([[0.0, 50.0, 100.0]], dtype=np.float32)
    rgb = compose_rgb_uint8([_plane(0, data, zmin=50.0, zmax=100.0)])
    # Only red tint for channel 0.
    assert rgb[0, 0, 0] == 0
    assert rgb[0, 1, 0] == 0
    assert rgb[0, 2, 0] == 255
    assert np.all(rgb[..., 1] == 0)
    assert np.all(rgb[..., 2] == 0)


def test_compose_rgb_explicit_tint_overrides_default() -> None:
    data = np.full((2, 2), 1.0, dtype=np.float32)
    rgb = compose_rgb_uint8(
        [_plane(0, data, zmin=0.0, zmax=1.0, tint_rgb=(0.0, 0.0, 1.0))]
    )
    assert np.all(rgb[..., 0] == 0)
    assert np.all(rgb[..., 1] == 0)
    assert np.all(rgb[..., 2] == 255)


def test_compose_rgb_samples_channel_colorscale_lut() -> None:
    """Chromatic colorscale drives composite color (not channel-index tint)."""
    from nicewidgets.contrast_widget.colorscales import get_colorscale

    data = np.full((2, 2), 1.0, dtype=np.float32)
    rgb = compose_rgb_uint8(
        [
            _plane(
                0,
                data,
                zmin=0.0,
                zmax=1.0,
                colorscale=get_colorscale('Magenta'),
            )
        ]
    )
    # Magenta LUT at full intensity is near-white-magenta (high R and B).
    assert int(rgb[0, 0, 0]) > 200
    assert int(rgb[0, 0, 1]) > 200
    assert int(rgb[0, 0, 2]) > 200
    # Must not collapse to pure red tint-by-channel-id.
    assert not (rgb[0, 0, 0] == 255 and rgb[0, 0, 1] == 0 and rgb[0, 0, 2] == 0)


def test_compose_rgb_clamps_additive_overflow() -> None:
    """Two full contributions into the same channel clamp at 255."""
    data = np.full((1, 1), 1.0, dtype=np.float32)
    rgb = compose_rgb_uint8(
        [
            _plane(0, data, zmin=0.0, zmax=1.0, tint_rgb=(1.0, 0.0, 0.0)),
            _plane(1, data, zmin=0.0, zmax=1.0, tint_rgb=(1.0, 0.0, 0.0)),
        ]
    )
    assert rgb[0, 0, 0] == 255


def test_compose_rgb_png_data_uri_roundtrip() -> None:
    data = np.full((4, 5), 1.0, dtype=np.float32)
    uri = compose_rgb_png_data_uri([_plane(0, data, zmin=0.0, zmax=1.0)])
    assert uri.startswith('data:image/png;base64,')
    raw = base64.b64decode(uri.split(',', 1)[1])
    with Image.open(io.BytesIO(raw)) as im:
        arr = np.asarray(im.convert('RGB'))
    assert arr.shape == (4, 5, 3)
    assert np.all(arr[..., 0] == 255)
    assert np.all(arr[..., 1] == 0)
    assert np.all(arr[..., 2] == 0)


def test_default_tint_for_channel_cycles_rgb() -> None:
    assert default_tint_for_channel(0) == (1.0, 0.0, 0.0)
    assert default_tint_for_channel(1) == (0.0, 1.0, 0.0)
    assert default_tint_for_channel(2) == (0.0, 0.0, 1.0)
    assert default_tint_for_channel(3) == (1.0, 0.0, 0.0)


def test_channel_plane_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match='2D'):
        ChannelPlane(channel_id=0, data=np.zeros((2, 2, 2), dtype=np.float32))


def test_compose_rgb_for_bounds_clips() -> None:
    data0 = np.zeros((4, 6), dtype=np.float32)
    data0[:, :] = 100.0
    data1 = np.zeros((4, 6), dtype=np.float32)
    data1[:, :] = 100.0
    rgb, used = compose_rgb_uint8_for_bounds(
        [
            _plane(0, data0, zmin=0.0, zmax=100.0),
            _plane(1, data1, zmin=0.0, zmax=100.0),
        ],
        RowColBounds(row_min=1, row_max=3, col_min=2, col_max=5),
    )
    assert rgb.shape == (2, 3, 3)
    assert used == RowColBounds(row_min=1.0, row_max=3.0, col_min=2.0, col_max=5.0)


def test_rgb_rows_cols_to_plotly_transposes() -> None:
    rgb = np.zeros((2, 5, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 2, 3)
    out = rgb_rows_cols_to_plotly_image_z(rgb)
    assert out.shape == (5, 2, 3)
    assert tuple(out[0, 0]) == (1, 2, 3)


def test_build_image_rgb_response_mode_and_orientation() -> None:
    grid = RasterGridSpec(dx=1.0, dy=2.0, x_unit='', y_unit='')
    planes = [
        _plane(0, np.full((4, 6), 100.0, dtype=np.float32), zmin=0.0, zmax=100.0),
        _plane(1, np.full((4, 6), 100.0, dtype=np.float32), zmin=0.0, zmax=100.0),
    ]
    response = build_image_rgb_response(
        planes,
        grid=grid,
        bounds=RowColBounds(row_min=0, row_max=4, col_min=0, col_max=6),
        max_pixels=None,
    )
    assert response.mode == 'image_rgb'
    assert response.rgb is not None
    assert response.rgb.shape == (6, 4, 3)
    assert response.rgb.dtype == np.uint8
    assert response.dx == 1.0
    assert response.dy == 2.0
