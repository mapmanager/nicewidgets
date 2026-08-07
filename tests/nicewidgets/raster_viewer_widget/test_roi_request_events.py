"""Unit tests for ROI request event payload parsing."""

from __future__ import annotations

from nicegui import events

from nicewidgets.raster_viewer_widget.events import (
    RasterRoiAddRequestedEvent,
    RasterRoiDeleteRequestedEvent,
    RasterRoiEditCancelRequestedEvent,
    RasterRoiEditRequestedEvent,
)
from nicewidgets.raster_viewer_widget.roi import RoiType


def _nicegui_event(payload: dict[str, object]) -> events.GenericEventArguments:
    """Build a minimal NiceGUI custom-event wrapper."""
    return events.GenericEventArguments(sender=None, client=None, args=payload)


def test_roi_add_requested_defaults_to_rect() -> None:
    """Parse add-request preferred type with rect default."""
    event = RasterRoiAddRequestedEvent.from_nicegui(
        _nicegui_event({"dataset_id": "ds", "preferred_type": "rectroi"})
    )
    assert event.dataset_id == "ds"
    assert event.preferred_type is RoiType.RECTROI


def test_roi_delete_and_edit_request_ids() -> None:
    """Parse delete and edit request ROI identities."""
    delete_event = RasterRoiDeleteRequestedEvent.from_nicegui(
        _nicegui_event({"dataset_id": "ds", "roi_id": 7})
    )
    edit_event = RasterRoiEditRequestedEvent.from_nicegui(
        _nicegui_event({"dataset_id": "ds", "roi_id": 8})
    )
    cancel_event = RasterRoiEditCancelRequestedEvent.from_nicegui(
        _nicegui_event({"dataset_id": "ds", "roi_id": None})
    )
    assert delete_event.roi_id == 7
    assert edit_event.roi_id == 8
    assert cancel_event.roi_id is None
