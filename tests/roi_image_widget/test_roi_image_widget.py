# nicewidgets/tests/roi_image_widget/test_roi_image_widget.py

from __future__ import annotations

import pytest

# Skip this entire module if matplotlib is not installed (optional dependency).
pytest.importorskip("matplotlib")

import numpy as np

from nicewidgets.roi_image_widget.viewport import (
    Viewport,
    full_to_view,
    view_to_full,
)
from nicewidgets.roi_image_widget.roi_image_widget import (
    RoiImageWidget,
    RoiImageConfig,
    RoiDict,
)


# ---------------------------------------------------------------------
# Viewport tests
# ---------------------------------------------------------------------


def test_viewport_roundtrip():
    """full_to_view followed by view_to_full should approximately recover the original point."""
    img_w, img_h = 200, 100
    vp = Viewport(img_width=img_w, img_height=img_h)

    disp_w, disp_h = 400, 200  # arbitrary display size

    # Pick a few points in full-image coordinates
    test_points = [
        (0.0, 0.0),
        (img_w / 2.0, img_h / 2.0),
        (img_w - 1.0, img_h - 1.0),
        (42.3, 17.7),
    ]

    for x, y in test_points:
        vx, vy = full_to_view(x, y, vp, disp_w, disp_h)
        x2, y2 = view_to_full(vx, vy, vp, disp_w, disp_h)
        assert x2 == pytest.approx(x, rel=1e-6, abs=1e-6)
        assert y2 == pytest.approx(y, rel=1e-6, abs=1e-6)

    # After a zoom, roundtrip should still be consistent
    vp.zoom_around(
        x_center=img_w / 2.0,
        y_center=img_h / 2.0,
        factor_x=0.5,
        factor_y=0.5,
    )
    for x, y in test_points:
        vx, vy = full_to_view(x, y, vp, disp_w, disp_h)
        x2, y2 = view_to_full(vx, vy, vp, disp_w, disp_h)
        assert x2 == pytest.approx(x, rel=1e-6, abs=1e-6)
        assert y2 == pytest.approx(y, rel=1e-6, abs=1e-6)


def test_viewport_pan_preserves_size_and_clamps():
    """Pan should preserve viewport width/height and clamp inside the image."""
    img_w, img_h = 200, 100
    vp = Viewport(img_width=img_w, img_height=img_h)

    # Start in full view
    w0 = vp.width
    h0 = vp.height
    assert w0 == pytest.approx(float(img_w))
    assert h0 == pytest.approx(float(img_h))

    # Pan by a large amount: should clamp at edges, but keep size
    vp.pan(dx=1000.0, dy=1000.0)
    assert vp.width == pytest.approx(w0)
    assert vp.height == pytest.approx(h0)
    assert 0.0 <= vp.x_min <= vp.x_max <= float(img_w)
    assert 0.0 <= vp.y_min <= vp.y_max <= float(img_h)

    # Pan in the opposite direction heavily; still clamped and same size
    vp.pan(dx=-1000.0, dy=-1000.0)
    assert vp.width == pytest.approx(w0)
    assert vp.height == pytest.approx(h0)
    assert 0.0 <= vp.x_min <= vp.x_max <= float(img_w)
    assert 0.0 <= vp.y_min <= vp.y_max <= float(img_h)


# ---------------------------------------------------------------------
# RoiImageWidget tests
# ---------------------------------------------------------------------


def _make_dummy_widget(shape=(50, 100), config: RoiImageConfig | None = None) -> RoiImageWidget:
    """Helper to create a small RoiImageWidget for tests."""
    img = np.zeros(shape, dtype=float)
    if config is None:
        config = RoiImageConfig()
    widget = RoiImageWidget(img, config=config)
    return widget


def test_set_rois_and_next_id():
    """set_rois with an existing id like 'roi-5' should bump _next_id so new ROIs get 'roi-6'."""
    img = np.zeros((50, 100), dtype=float)

    # Simple config; we don't care about visuals for this test.
    config = RoiImageConfig()

    widget = RoiImageWidget(img, config=config)

    rois: list[RoiDict] = [
        {"id": "roi-5", "left": 10, "top": 5, "right": 30, "bottom": 25},
    ]
    widget.set_rois(rois)

    # Use the internal API to create a new ROI and check its id.
    new_roi = widget._create_new_roi(40.0, 10.0)
    assert new_roi.id == "roi-6"


def test_get_rois_roundtrip():
    """get_rois should return the same ROIs (up to int rounding) that were set."""
    widget = _make_dummy_widget()

    rois_in: list[RoiDict] = [
        {"id": "roi-1", "left": 5, "top": 3, "right": 25, "bottom": 15},
        {"id": "roi-2", "left": 30, "top": 10, "right": 45, "bottom": 20},
    ]
    widget.set_rois(rois_in)

    rois_out = widget.get_rois()
    # Convert to dict keyed by id for easier comparison
    out_by_id = {r["id"]: r for r in rois_out}

    assert set(out_by_id.keys()) == {"roi-1", "roi-2"}
    for r in rois_in:
        rid = r["id"]
        assert rid in out_by_id
        # They should match exactly as ints
        assert out_by_id[rid]["left"] == r["left"]
        assert out_by_id[rid]["top"] == r["top"]
        assert out_by_id[rid]["right"] == r["right"]
        assert out_by_id[rid]["bottom"] == r["bottom"]


def test_hit_test_edges():
    """_hit_test should detect edge hits and distinguish left/right/top/bottom."""
    img = np.zeros((100, 200), dtype=float)

    config = RoiImageConfig(edge_tolerance_px=5.0)
    widget = RoiImageWidget(img, config=config)

    # Single ROI in the middle of the image
    roi_dict: RoiDict = {"id": "roi-1", "left": 50, "top": 20, "right": 150, "bottom": 80}
    widget.set_rois([roi_dict])

    # Because viewport is full-image and display size == image size by default,
    # full coords and display coords are 1:1 for this test.

    # Left edge: x at 50, y in the vertical middle
    roi_id, mode = widget._hit_test(50.0, 50.0)
    assert roi_id == "roi-1"
    assert mode == "resizing_left"

    # Right edge: x at 150, y in the vertical middle
    roi_id, mode = widget._hit_test(150.0, 50.0)
    assert roi_id == "roi-1"
    assert mode == "resizing_right"

    # Top edge: y at 20, x in the horizontal middle
    roi_id, mode = widget._hit_test(100.0, 20.0)
    assert roi_id == "roi-1"
    assert mode == "resizing_top"

    # Bottom edge: y at 80, x in the horizontal middle
    roi_id, mode = widget._hit_test(100.0, 80.0)
    assert roi_id == "roi-1"
    assert mode == "resizing_bottom"

    # Inside but away from edges -> body
    roi_id, mode = widget._hit_test(100.0, 50.0)
    assert roi_id == "roi-1"
    assert mode == "body"

    # Outside -> no hit
    roi_id, mode = widget._hit_test(10.0, 10.0)
    assert roi_id is None
    assert mode is None


def test_dtype_and_contrast_api():
    """dtype property and contrast methods should behave sensibly and not crash."""
    # Gradient image to have a non-trivial range
    height, width = 20, 40
    img = np.linspace(0.0, 1.0, height * width, dtype=float).reshape(height, width)

    widget = RoiImageWidget(img, config=RoiImageConfig())

    # dtype should reflect underlying numpy dtype
    assert widget.dtype == img.dtype

    # set_contrast(None, None) should reset to full range
    widget.set_contrast(None, None)
    # Access internal vmin/vmax for sanity check (we already use internal APIs elsewhere)
    assert widget._vmin <= 0.0 + 1e-6
    assert widget._vmax >= 1.0 - 1e-6

    # auto_contrast should set a narrower range but still within [0, 1]
    widget.auto_contrast(low_percentile=10.0, high_percentile=90.0)
    assert 0.0 <= widget._vmin < widget._vmax <= 1.0

    # And rendering with the new contrast should not raise
    img_pil = widget._render_view_pil()
    assert img_pil.size[0] == widget.DISPLAY_W
    assert img_pil.size[1] == widget.DISPLAY_H
