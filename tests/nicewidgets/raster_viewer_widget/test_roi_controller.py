"""Tests for delegated JS ROI chrome demo coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

from examples.raster_viewer_widget.roi_controller import DemoRoiController
from examples.raster_viewer_widget.roi_store import DemoRoiStore
from nicewidgets.raster_viewer_widget.events import (
    RasterRoiAddRequestedEvent,
    RasterRoiDeleteRequestedEvent,
    RasterRoiEditCancelRequestedEvent,
    RasterRoiEditCommittedEvent,
    RasterRoiEditRequestedEvent,
    RasterRoiSelectedEvent,
    RasterRoiStateChangeEvent,
)
from nicewidgets.raster_viewer_widget.roi import ImageBounds, LineEndpoints, RectRoiBounds


class _DatasetProvider:
    """Provide one deterministic mixed-shape store to the controller."""

    def __init__(self) -> None:
        self.store = DemoRoiStore(ImageBounds(width=100, height=80))
        self.first_rect = self.store.create_rect(RectRoiBounds(10, 30, 20, 50))
        self.line = self.store.create_line(LineEndpoints(5, 6, 30, 40))

    def get_roi_store(self, dataset_id: str) -> DemoRoiStore:
        """Return the test store."""
        assert dataset_id == "dataset"
        return self.store

    def channel_indices(self, dataset_id: str) -> tuple[int, ...]:
        """Return two logical channels."""
        assert dataset_id == "dataset"
        return (0, 1)


class _Selector:
    """Record whether dataset interaction is enabled."""

    def __init__(self) -> None:
        self.enabled = True

    def set_enabled(self, enabled: bool) -> None:
        """Record the current enabled state."""
        self.enabled = enabled


class _RoiCommands:
    """Record controller calls to the namespaced viewer ROI API."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def add(self, roi: object) -> bool:
        """Record an add command."""
        self.calls.append(("add", roi))
        return True

    async def remove(self, roi_id: int) -> bool:
        """Record a remove command."""
        self.calls.append(("remove", roi_id))
        return True

    async def select(self, roi_id: int | None) -> bool:
        """Record a selection command."""
        self.calls.append(("select", roi_id))
        return True

    async def begin_edit(self, roi_id: int) -> bool:
        """Record an edit command."""
        self.calls.append(("begin_edit", roi_id))
        return True

    async def commit_edit(self) -> bool:
        """Record a commit request."""
        self.calls.append(("commit_edit", True))
        return True

    async def cancel_edit(self) -> bool:
        """Record a cancellation request."""
        self.calls.append(("cancel_edit", True))
        return True

    async def complete_commit(self, roi: object) -> bool:
        """Record an authoritative commit completion."""
        self.calls.append(("complete_commit", roi))
        return True


class _Viewer:
    """Expose callback registration and fake ROI commands."""

    def __init__(self) -> None:
        self.rois = _RoiCommands()
        self.selected_handler: Callable[[RasterRoiSelectedEvent], Any] | None = None
        self.add_handler: Callable[[RasterRoiAddRequestedEvent], Any] | None = None
        self.delete_handler: Callable[[RasterRoiDeleteRequestedEvent], Any] | None = None
        self.edit_handler: Callable[[RasterRoiEditRequestedEvent], Any] | None = None
        self.cancel_handler: Callable[[RasterRoiEditCancelRequestedEvent], Any] | None = None
        self.state_handler: Callable[[RasterRoiStateChangeEvent], Any] | None = None
        self.commit_handler: Callable[[RasterRoiEditCommittedEvent], Any] | None = None

    def on_roi_selected(self, handler: Callable[[RasterRoiSelectedEvent], Any]) -> object:
        """Store the selection handler."""
        self.selected_handler = handler
        return self

    def on_roi_add_requested(
        self, handler: Callable[[RasterRoiAddRequestedEvent], Any]
    ) -> object:
        """Store the add-request handler."""
        self.add_handler = handler
        return self

    def on_roi_delete_requested(
        self, handler: Callable[[RasterRoiDeleteRequestedEvent], Any]
    ) -> object:
        """Store the delete-request handler."""
        self.delete_handler = handler
        return self

    def on_roi_edit_requested(
        self, handler: Callable[[RasterRoiEditRequestedEvent], Any]
    ) -> object:
        """Store the edit-request handler."""
        self.edit_handler = handler
        return self

    def on_roi_edit_cancel_requested(
        self, handler: Callable[[RasterRoiEditCancelRequestedEvent], Any]
    ) -> object:
        """Store the edit-cancel handler."""
        self.cancel_handler = handler
        return self

    def on_roi_state_change(
        self, handler: Callable[[RasterRoiStateChangeEvent], Any]
    ) -> object:
        """Store the state handler."""
        self.state_handler = handler
        return self

    def on_roi_edit_committed(
        self, handler: Callable[[RasterRoiEditCommittedEvent], Any]
    ) -> object:
        """Store the edit-commit handler."""
        self.commit_handler = handler
        return self


def _controller() -> tuple[DemoRoiController, _DatasetProvider, _Viewer]:
    """Build one controller with deterministic collaborators."""
    datasets = _DatasetProvider()
    viewer = _Viewer()
    controller = DemoRoiController(
        datasets,
        cast(Any, _Selector()),
        cast(Any, viewer),
        "dataset",
    )
    return controller, datasets, viewer


def test_controller_tracks_rectangle_selection_not_lines() -> None:
    """Keep line ROIs out of rectangle Add/Delete/Edit coordination."""
    controller, datasets, _viewer = _controller()
    assert controller.selected_roi_id == datasets.first_rect.roi_id
    assert datasets.first_rect.roi_id in [
        roi.roi_id for roi in datasets.store if hasattr(roi, "bounds")
    ]


def test_add_and_delete_are_immediate_rectangle_operations() -> None:
    """Match instant-add / delete semantics for delegated ROI chrome requests."""
    controller, datasets, viewer = _controller()

    async def exercise() -> None:
        await controller._add_rectangle()
        added_id = controller.selected_roi_id
        assert added_id is not None
        assert added_id != datasets.first_rect.roi_id
        await controller._delete_rectangle(added_id)

    asyncio.run(exercise())
    assert [name for name, _value in viewer.rois.calls] == [
        "add",
        "select",
        "remove",
        "select",
    ]
    assert controller.selected_roi_id == datasets.first_rect.roi_id
