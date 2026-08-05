"""Tests for discoverable typed raster-viewer events."""

from typing import Any, cast

from nicegui import events

from nicewidgets.raster_viewer_widget import LineRoiCreate, RoiInteractionState
from nicewidgets.raster_viewer_widget.events import (
    RasterChannelSelectedEvent,
    RasterErrorEvent,
    RasterPlaneChangeEvent,
    RasterReadyEvent,
    RasterRoiCreateRequestedEvent,
    RasterRoiEditCommittedEvent,
    RasterRoiStateChangeEvent,
    RasterViewChangeEvent,
)


def _generic_event(payload: dict[str, Any]) -> events.GenericEventArguments:
    """Create a minimal NiceGUI event wrapper for conversion tests.

    Args:
        payload: Custom-event detail mapping.

    Returns:
        Generic event carrying the mapping.
    """
    return cast(events.GenericEventArguments, type("Event", (), {"args": payload})())


def test_ready_event_exposes_dataset_and_axis_properties() -> None:
    """Verify typed ready events retain canonical payload values."""
    event = RasterReadyEvent.from_nicegui(
        _generic_event(
            {
                "dataset_id": "sample",
                "x_axis": {"minimum": 0.0, "maximum": 10.0},
            }
        )
    )
    assert isinstance(event, RasterReadyEvent)
    assert event.dataset_id == "sample"
    assert event.x_axis.maximum == 10.0


def test_error_event_has_safe_human_readable_message() -> None:
    """Verify typed errors expose explicit and fallback messages."""
    assert RasterErrorEvent.from_nicegui(_generic_event({"message": "failed"})).message == "failed"
    assert "Unknown" in RasterErrorEvent.from_nicegui(_generic_event({})).message


def test_channel_and_t_z_events_are_typed() -> None:
    """Verify selection and multidimensional plane callbacks expose named values."""
    selected = RasterChannelSelectedEvent.from_nicegui(
        _generic_event({"dataset_id": "sample", "channel_id": "channel_1"})
    )
    assert selected.channel_id == "channel_1"
    plane = RasterPlaneChangeEvent.from_nicegui(
        _generic_event({"t_index": 2, "z_index": 4, "plus_minus_z": 1})
    )
    assert plane.t_index == 2
    assert plane.z_index == 4
    assert plane.plus_minus_z == 1


def test_view_event_exposes_physical_axis_ranges() -> None:
    """Verify linked views can consume physical ranges without JS interpretation."""
    event = RasterViewChangeEvent.from_nicegui(_generic_event({
        "cause": "wheel",
        "final": True,
        "physical_range": {
            "x": {"minimum": 1.0, "maximum": 3.0, "label": "s", "unit": ""},
            "y": {"minimum": 2.0, "maximum": 6.0, "label": "um", "unit": ""},
        },
    }))
    assert event.x_range.maximum == 3.0
    assert event.y_range.minimum == 2.0


def test_roi_events_convert_line_geometry_to_typed_values() -> None:
    """Keep browser JSON parsing at the widget event boundary."""
    creation = RasterRoiCreateRequestedEvent.from_nicegui(_generic_event({
        "dataset_id": "sample",
        "roi_type": "linesegmentroi",
        "name": "2",
        "note": "",
        "data": {"row0": 1, "col0": 2, "row1": 7, "col1": 9},
    }))
    assert isinstance(creation.specification, LineRoiCreate)
    edited = RasterRoiEditCommittedEvent.from_nicegui(_generic_event({
        "roi": {
            "roi_id": 2,
            "roi_type": "linesegmentroi",
            "version": "1.0",
            "name": "1",
            "note": "",
            "data": {"row0": 1, "col0": 2, "row1": 7, "col1": 9},
        }
    }))
    assert edited.roi.roi_type.value == "linesegmentroi"
    state = RasterRoiStateChangeEvent.from_nicegui(_generic_event({"state": "editing"}))
    assert state.state is RoiInteractionState.EDITING
