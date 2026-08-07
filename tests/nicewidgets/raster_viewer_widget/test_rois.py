"""Tests for reusable mixed-shape ROI contracts and demo ownership."""

import pytest

from examples.raster_viewer_widget.roi_store import DemoRoiStore
from nicewidgets.raster_viewer_widget import (
    ImageBounds,
    LineEndpoints,
    LineRoi,
    RectRoi,
    RectRoiBounds,
    roi_from_mapping,
)


def test_rect_bounds_normalize_clamp_and_remain_nonempty() -> None:
    """Verify inverted and out-of-image edges become a valid rectangle."""
    bounds = RectRoiBounds(20, -4, 99, -2).clamped_to(ImageBounds(10, 8))
    assert bounds == RectRoiBounds(0, 8, 0, 10)


def test_line_endpoints_clamp_as_pixel_indices_without_reordering() -> None:
    """Keep endpoint identity while constraining actual pixel coordinates."""
    endpoints = LineEndpoints(99, -2, -4, 20).clamped_to(ImageBounds(10, 8))
    assert endpoints == LineEndpoints(7, 0, 0, 9)


def test_mixed_roi_envelopes_round_trip_strictly() -> None:
    """Serialize and parse both supported shape discriminators."""
    rois = (
        RectRoi(1, "0", RectRoiBounds(1, 5, 2, 7)),
        LineRoi(2, "1", LineEndpoints(1, 2, 7, 9)),
    )
    assert tuple(roi_from_mapping(roi.to_json()) for roi in rois) == rois
    assert rois[1].to_json()["roi_type"] == "linesegmentroi"


def test_empty_roi_name_is_allowed_identity_is_roi_id() -> None:
    """Optional display names may be blank; roi_id remains required and positive."""
    roi = RectRoi(4, "", RectRoiBounds(1, 5, 2, 7))
    assert roi.name == ""
    assert roi_from_mapping(roi.to_json()) == roi
    with pytest.raises(ValueError, match="roi_id must be positive"):
        RectRoi(0, "", RectRoiBounds(1, 5, 2, 7))


def test_store_uses_one_monotonic_identity_namespace_for_mixed_shapes() -> None:
    """Deleting either shape does not reuse IDs or numeric display names."""
    store = DemoRoiStore(ImageBounds(100, 80))
    first = store.create_rect(RectRoiBounds(1, 10, 2, 20))
    second = store.create_line(LineEndpoints(10, 20, 30, 40))
    store.delete(first.roi_id)
    third = store.create_rect(RectRoiBounds(30, 50, 40, 60))
    assert (first.roi_id, second.roi_id, third.roi_id) == (1, 2, 3)
    assert (first.name, second.name, third.name) == ("0", "1", "2")


def test_store_update_clamps_and_preserves_authoritative_metadata() -> None:
    """Mixed-shape committed edits remain bounded and preserve identity."""
    store = DemoRoiStore(ImageBounds(20, 10))
    original = store.create_line(LineEndpoints(1, 2, 5, 7), note="owner")
    edited = store.update(LineRoi(original.roi_id, "ignored", LineEndpoints(-5, -8, 100, 200)))
    assert edited == LineRoi(original.roi_id, original.name, LineEndpoints(0, 0, 9, 19), "owner")


def test_unknown_update_and_delete_raise() -> None:
    """Verify invalid stable IDs fail explicitly."""
    store = DemoRoiStore(ImageBounds(20, 10))
    with pytest.raises(KeyError):
        store.update(RectRoi(9, "9", RectRoiBounds(1, 2, 3, 4)))
    with pytest.raises(KeyError):
        store.delete(9)


def test_suggested_creation_geometry_is_centered_and_bounded() -> None:
    """Provide deterministic drafts for both demo Add choices."""
    store = DemoRoiStore(ImageBounds(100, 80))
    assert store.suggested_rect_bounds() == RectRoiBounds(30, 50, 37, 62)
    assert store.suggested_line_endpoints() == LineEndpoints(39, 37, 39, 62)
