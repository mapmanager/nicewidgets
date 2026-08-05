"""Tests for reusable NumPy raster sources and descriptors."""

import numpy as np
import pytest

from nicewidgets.raster_viewer_widget.models import RasterPlaneRequest
from nicewidgets.raster_viewer_widget.numpy_source import NumPyRasterSource
from nicewidgets.raster_viewer_widget.roi import LineEndpoints, LineRoi


def test_from_array_splits_named_channel_axis_and_excludes_it_from_header() -> None:
    """Verify one Z/C/Y/X array becomes logical same-shaped channels."""
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    source = NumPyRasterSource.from_array(
        data,
        dims=("Z", "C", "Y", "X"),
        physical_units=(1.0, 1.0, 0.2, 0.3),
        physical_units_labels=("slice", "channel", "um", "um"),
        channel_ids=("green", "red"),
    )
    descriptor = source.get_descriptor()
    assert descriptor.header.shape == (3, 4, 5)
    assert descriptor.header.dims == ("Z", "Y", "X")
    assert descriptor.header.num_channels == 2
    assert [channel.channel_id for channel in descriptor.channels] == ["green", "red"]
    np.testing.assert_array_equal(
        source.get_plane(RasterPlaneRequest("red", z_index=2)),
        data[2, 1],
    )


def test_from_array_does_not_eagerly_copy_each_channel_volume() -> None:
    """Verify a named C axis remains backed by the caller's contiguous array."""
    data = np.zeros((3, 2, 4, 5), dtype=np.uint16)
    source = NumPyRasterSource.from_array(
        data,
        dims=("Z", "C", "Y", "X"),
        physical_units=(1.0, 1.0, 1.0, 1.0),
        physical_units_labels=("slice", "channel", "px", "px"),
    )
    assert all(
        np.shares_memory(channel.values, data)  # noqa: SLF001
        for channel in source._channels.values()  # noqa: SLF001
    )


def test_from_array_without_channel_axis_creates_one_channel() -> None:
    """Verify a plain Y/X array remains a single logical channel."""
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    source = NumPyRasterSource.from_array(
        data,
        dims=("Y", "X"),
        physical_units=(0.5, 0.25),
        physical_units_labels=("um", "um"),
    )
    assert source.get_descriptor().header.num_channels == 1
    np.testing.assert_array_equal(
        source.get_plane(RasterPlaneRequest("channel_0")), data
    )


def test_from_array_serializes_typed_initial_rois() -> None:
    """Allow callers to pass reusable ROI values without manual JSON mappings."""
    source = NumPyRasterSource.from_array(
        np.zeros((4, 5), dtype=np.uint16),
        dims=("Y", "X"),
        physical_units=(1.0, 1.0),
        physical_units_labels=("px", "px"),
        rois=(LineRoi(1, "0", LineEndpoints(0, 0, 3, 4)),),
    )
    assert source.get_descriptor().rois[0]["roi_type"] == "linesegmentroi"


def test_from_array_rejects_ambiguous_or_unsupported_dimensions() -> None:
    """Verify explicit dimension metadata fails early and clearly."""
    data = np.zeros((2, 4, 5), dtype=np.uint16)
    with pytest.raises(ValueError, match="unique"):
        NumPyRasterSource.from_array(
            data,
            dims=("Y", "Y", "X"),
            physical_units=(1.0, 1.0, 1.0),
            physical_units_labels=("px", "px", "px"),
        )
    with pytest.raises(ValueError, match="dims may contain"):
        NumPyRasterSource.from_array(
            data,
            dims=("Q", "Y", "X"),
            physical_units=(1.0, 1.0, 1.0),
            physical_units_labels=("q", "px", "px"),
        )
def test_sliding_projection_is_clamped_to_available_planes() -> None:
    """Verify centered projection handles stack edges without padding."""
    data = np.stack(
        [np.full((2, 3), value, dtype=np.uint16) for value in (2, 7, 4)]
    )
    source = NumPyRasterSource.from_array(
        data,
        dims=("Z", "Y", "X"),
        physical_units=(1.0, 1.0, 1.0),
        physical_units_labels=("slice", "px", "px"),
    )
    plane = source.get_plane(
        RasterPlaneRequest("channel_0", z_index=0, plus_minus_z=1)
    )
    np.testing.assert_array_equal(plane, np.full((2, 3), 7, dtype=np.uint16))


def test_t_z_source_selects_named_axes_and_projects_z_at_fixed_t() -> None:
    """Verify T/Z extraction uses names rather than assuming one axis order."""
    data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    source = NumPyRasterSource.from_array(
        data,
        dims=("T", "Z", "Y", "X"),
        physical_units=(0.5, 1.0, 0.2, 0.3),
        physical_units_labels=("s", "slice", "um", "um"),
    )
    plane = source.get_plane(RasterPlaneRequest("channel_0", t_index=1, z_index=1))
    np.testing.assert_array_equal(plane, data[1, 1])
    projected = source.get_plane(
        RasterPlaneRequest("channel_0", t_index=0, z_index=1, plus_minus_z=1)
    )
    np.testing.assert_array_equal(projected, np.max(data[0], axis=0))


def test_z_t_source_preserves_descriptor_order_but_indexes_by_name() -> None:
    """Verify the supported leading axes may appear in either order."""
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    source = NumPyRasterSource.from_array(
        data,
        dims=("Z", "T", "Y", "X"),
        physical_units=(1.0, 0.5, 0.2, 0.3),
        physical_units_labels=("slice", "s", "um", "um"),
    )
    plane = source.get_plane(RasterPlaneRequest("channel_0", t_index=1, z_index=2))
    np.testing.assert_array_equal(plane, data[2, 1])


@pytest.mark.parametrize(
    "arguments",
    [
        {"channel_id": " "},
        {"channel_id": "channel_0", "t_index": -1},
        {"channel_id": "channel_0", "z_index": -1},
        {"channel_id": "channel_0", "plus_minus_z": -1},
    ],
)
def test_plane_request_rejects_invalid_identity_or_negative_selection(
    arguments: dict[str, object],
) -> None:
    """Verify transport requests fail before invoking a source."""
    with pytest.raises(ValueError):
        RasterPlaneRequest(**arguments)  # type: ignore[arg-type]
