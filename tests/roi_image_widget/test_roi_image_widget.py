# nicewidgets/tests/roi_image_widget/test_roi_image_widget.py

from __future__ import annotations

import numpy as np
import pytest

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
