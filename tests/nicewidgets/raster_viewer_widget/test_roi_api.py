"""Tests for the namespaced typed Python ROI API."""

from __future__ import annotations

import asyncio
from typing import Any

from nicewidgets.raster_viewer_widget import (
    LineEndpoints,
    LineRoi,
    LineRoiCreate,
    RectRoi,
    RectRoiBounds,
)
from nicewidgets.raster_viewer_widget.roi_api import RoiApi
from nicewidgets.raster_viewer_widget.widget import RasterViewerWidget


def test_namespaced_roi_api_routes_typed_mixed_shape_operations() -> None:
    """Keep the public namespace typed while the component bridge stays flat."""
    calls: list[tuple[str, tuple[object, ...]]] = []

    async def run_method(method: str, *arguments: object) -> Any:
        calls.append((method, arguments))
        return 2 if method == "setRois" else True

    async def exercise() -> None:
        api = RoiApi(run_method)
        rect = RectRoi(1, "0", RectRoiBounds(1, 5, 2, 7))
        line = LineRoi(2, "1", LineEndpoints(1, 2, 7, 9))
        assert await api.set((rect, line)) == 2
        assert await api.add(line)
        assert await api.update(rect)
        assert await api.select(2)
        assert await api.clear_selection()
        assert await api.begin_create(LineRoiCreate("2", LineEndpoints(2, 3, 5, 8)))
        assert await api.begin_edit(2)
        assert await api.commit_edit()
        assert await api.cancel_edit()
        assert await api.complete_commit(line)
        assert await api.remove(2)

    asyncio.run(exercise())
    assert [method for method, _arguments in calls] == [
        "setRois", "addRoi", "updateRoi", "selectRoi", "selectRoi",
        "beginRoiCreate", "beginRoiEdit", "commitRoiEdit", "cancelRoiEdit",
        "completeRoiCommit", "removeRoi",
    ]


def test_widget_does_not_retain_flat_roi_compatibility_methods() -> None:
    """Keep ROI organization exclusively under the public namespace."""
    for name in (
        "set_rois", "add_roi", "update_roi", "remove_roi", "select_roi",
        "begin_roi_create", "begin_roi_edit", "commit_roi_edit",
        "cancel_roi_edit", "complete_roi_commit",
    ):
        assert not hasattr(RasterViewerWidget, name)
