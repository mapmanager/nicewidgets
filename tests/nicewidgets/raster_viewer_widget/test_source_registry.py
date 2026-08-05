"""Tests for independent per-widget source registrations."""

import numpy as np
import pytest

from nicewidgets.raster_viewer_widget.models import RasterChannelDisplay
from nicewidgets.raster_viewer_widget.numpy_source import NumPyRasterSource
from nicewidgets.raster_viewer_widget.source_registry import (
    REGISTRY,
    RasterSourceRegistry,
    _descriptor_json,
    source_plane,
)


def _source(source_id: str) -> NumPyRasterSource:
    """Return a compact test source.

    Args:
        source_id: Stable descriptor identity.

    Returns:
        One-channel NumPy source.
    """
    return NumPyRasterSource.from_array(
        np.zeros((3, 4), dtype=np.uint16),
        dims=("Y", "X"),
        physical_units=(1.0, 1.0),
        physical_units_labels=("px", "px"),
        source_id=source_id,
    )


def test_two_widgets_receive_independent_tokens_for_the_same_source() -> None:
    """Verify registrations isolate instances even when Python data is shared."""
    registry = RasterSourceRegistry()
    source = _source("shared")
    first = registry.register(source)
    second = registry.register(source)
    assert first != second
    assert registry.get(first) is source
    assert registry.get(second) is source


def test_unregistering_one_widget_preserves_the_other() -> None:
    """Verify destroying one instance cannot release its neighbor's source."""
    registry = RasterSourceRegistry()
    first_source = _source("first")
    second_source = _source("second")
    first = registry.register(first_source)
    second = registry.register(second_source)
    registry.unregister(first)
    with pytest.raises(KeyError):
        registry.get(first)
    assert registry.get(second) is second_source


def test_transport_descriptor_declares_exact_schema_and_binary_contract() -> None:
    """Verify browser metadata is versioned and matches a uint16 plane."""
    descriptor = _descriptor_json("opaque", _source("sample").get_descriptor())
    assert descriptor["schema_version"] == "2.0"
    assert descriptor["layout"] == "row-major"
    assert descriptor["endianness"] == "little"
    channel = descriptor["channels"][0]
    assert channel["encoding"] == "raw-u16-le"
    assert channel["byte_length"] == 3 * 4 * 2


def test_transport_descriptor_contains_atomic_initial_channel_display() -> None:
    """Verify source loading carries persisted display state in one descriptor."""
    source = NumPyRasterSource.from_array(
        np.zeros((3, 4), dtype=np.uint16),
        dims=("Y", "X"),
        physical_units=(1.0, 1.0),
        physical_units_labels=("px", "px"),
        channel_displays=(RasterChannelDisplay("green", 2.0, 8.0, False),),
    )
    descriptor = _descriptor_json("opaque", source.get_descriptor())
    assert descriptor["channels"][0]["display"] == {
        "lut": "green",
        "value_min": 2.0,
        "value_max": 8.0,
        "visible": False,
    }


def test_binary_endpoint_selects_t_z_and_projects_only_z() -> None:
    """Verify endpoint query parameters produce one fixed-T sliding-Z plane."""
    values = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
    source = NumPyRasterSource.from_array(
        values,
        dims=("T", "Z", "Y", "X"),
        physical_units=(1.0, 1.0, 1.0, 1.0),
        physical_units_labels=("s", "slice", "px", "px"),
    )
    token = REGISTRY.register(source)
    try:
        response = source_plane(
            token,
            "channel_0",
            t_index=1,
            z_index=1,
            plus_minus_z=1,
        )
    finally:
        REGISTRY.unregister(token)
    actual = np.frombuffer(response.body, dtype="<u2").reshape(4, 5)
    np.testing.assert_array_equal(actual, np.max(values[1], axis=0))
    assert response.headers["content-encoding"] == "identity"
