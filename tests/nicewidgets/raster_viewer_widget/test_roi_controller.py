"""Tests for rectangle-only ImageToolbar demo coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, cast

from examples.raster_viewer_widget.roi_controller import DemoRoiController
from examples.raster_viewer_widget.roi_store import DemoRoiStore
from nicewidgets.raster_viewer_widget.events import (
    RasterRoiEditCommittedEvent,
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
        self.state_handler: Callable[[RasterRoiStateChangeEvent], Any] | None = None
        self.edit_handler: Callable[[RasterRoiEditCommittedEvent], Any] | None = None

    def on_roi_selected(self, handler: Callable[[RasterRoiSelectedEvent], Any]) -> object:
        """Store the selection handler."""
        self.selected_handler = handler
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
        """Store the edit handler."""
        self.edit_handler = handler
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


def test_toolbar_lists_channels_but_only_rectangle_rois() -> None:
    """Keep line ROIs out of the rectangle-specific toolbar contract."""
    controller, datasets, _viewer = _controller()
    assert controller._toolbar.get_channel_options() == ["0", "1"]
    assert controller._toolbar.get_roi_options() == [datasets.first_rect.roi_id]


def test_add_and_delete_are_immediate_rectangle_operations() -> None:
    """Match ImageToolbarWidget's immediate Add/Delete intent semantics."""
    controller, datasets, viewer = _controller()

    async def exercise() -> None:
        await controller._add_rectangle()
        added_id = controller._toolbar.get_roi_id()
        assert added_id is not None
        assert controller._toolbar.get_roi_options() == [datasets.first_rect.roi_id, added_id]
        await controller._delete_rectangle(added_id)

    asyncio.run(exercise())
    assert [name for name, _value in viewer.rois.calls] == [
        "add",
        "select",
        "remove",
        "select",
    ]
    assert controller._toolbar.get_roi_options() == [datasets.first_rect.roi_id]
