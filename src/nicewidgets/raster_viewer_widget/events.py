"""Typed event values emitted by the NiceGUI raster viewer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self

from nicegui import events

from .roi import (
    LineEndpoints,
    LineRoiCreate,
    RectRoiBounds,
    RectRoiCreate,
    Roi,
    RoiCreate,
    RoiInteractionState,
    RoiType,
    roi_from_mapping,
)

type RasterEventHandler = Callable[[events.GenericEventArguments], Any]

READY_EVENT = "raster-ready"
ERROR_EVENT = "raster-error"
VIEW_CHANGE_EVENT = "raster-view-change"
DISPLAY_CHANGE_EVENT = "raster-display-change"
CHANNEL_SELECTED_EVENT = "raster-channel-selected"
TOOLBAR_ACTION_EVENT = "raster-toolbar-action"
PLANE_CHANGE_EVENT = "raster-plane-change"
PERFORMANCE_EVENT = "raster-performance"
ROI_SELECTED_EVENT = "raster-roi-select"
ROI_CREATE_REQUESTED_EVENT = "raster-roi-create"
ROI_EDIT_COMMITTED_EVENT = "raster-roi-edit-commit"
ROI_STATE_CHANGE_EVENT = "raster-roi-state-change"


@dataclass(frozen=True, slots=True)
class RasterAxisRange:
    """Typed physical axis range reported when a viewer becomes ready."""

    minimum: float
    maximum: float
    label: str = ""
    unit: str = ""

    @classmethod
    def from_mapping(cls, value: object) -> RasterAxisRange:
        """Validate and convert a JavaScript axis-range object."""
        if not isinstance(value, Mapping):
            raise ValueError("x_axis must be a mapping")
        return cls(
            minimum=float(value["minimum"]),
            maximum=float(value["maximum"]),
            label=str(value.get("label", "")),
            unit=str(value.get("unit", "")),
        )


@dataclass(frozen=True, slots=True)
class RasterEvent:
    """Store one canonical snake-case JavaScript event payload.

    Attributes:
        payload: Read-only-by-convention event data copied from NiceGUI.
    """

    payload: Mapping[str, Any]

    @classmethod
    def from_nicegui(cls, event: events.GenericEventArguments) -> Self:
        """Convert NiceGUI's generic event wrapper into a typed value.

        Args:
            event: NiceGUI custom-event wrapper.

        Returns:
            Typed raster event with an isolated payload dictionary.
        """
        payload = event.args if isinstance(event.args, dict) else {}
        return cls(dict(payload))

    @property
    def dataset_id(self) -> str | None:
        """Return the source dataset ID when the event carries one."""
        value = self.payload.get("dataset_id")
        return str(value) if value is not None else None


class RasterReadyEvent(RasterEvent):
    """Report completion of initial descriptor and plane loading."""

    @property
    def x_axis(self) -> RasterAxisRange:
        """Return the full physical display-X range."""
        return RasterAxisRange.from_mapping(self.payload.get("x_axis"))


class RasterErrorEvent(RasterEvent):
    """Report a component, transport, or viewer failure."""

    @property
    def message(self) -> str:
        """Return the human-readable error message."""
        return str(self.payload.get("message", "Unknown raster viewer error"))


class RasterViewChangeEvent(RasterEvent):
    """Report a user-originated viewport transformation."""

    @property
    def cause(self) -> str:
        """Return the stable interaction cause, such as wheel or drag zoom."""
        return str(self.payload.get("cause", ""))

    @property
    def final(self) -> bool:
        """Return whether this value ends the current interaction sequence."""
        return bool(self.payload.get("final", False))

    @property
    def channels(self) -> tuple[str, ...]:
        """Return channel IDs represented by the changed viewport."""
        value = self.payload.get("channels", ())
        return tuple(str(item) for item in value) if isinstance(value, list) else ()

    @property
    def x_range(self) -> RasterAxisRange:
        """Return the visible physical display-X range."""
        physical = self.payload.get("physical_range")
        if not isinstance(physical, Mapping):
            raise ValueError("physical_range must be a mapping")
        return RasterAxisRange.from_mapping(physical.get("x"))

    @property
    def y_range(self) -> RasterAxisRange:
        """Return the visible physical display-Y range."""
        physical = self.payload.get("physical_range")
        if not isinstance(physical, Mapping):
            raise ValueError("physical_range must be a mapping")
        return RasterAxisRange.from_mapping(physical.get("y"))


class RasterDisplayChangeEvent(RasterEvent):
    """Report LUT, range, visibility, or layout display state."""

    @property
    def cause(self) -> str:
        """Return ``user`` for browser edits or a non-user rendering cause."""
        return str(self.payload.get("cause", ""))

    @property
    def channels(self) -> tuple[Mapping[str, Any], ...]:
        """Return immutable-by-convention channel display snapshots."""
        value = self.payload.get("channels", ())
        if not isinstance(value, list):
            return ()
        return tuple(dict(item) for item in value if isinstance(item, Mapping))


class RasterChannelSelectedEvent(RasterEvent):
    """Report a user-originated active-channel selection."""

    @property
    def channel_id(self) -> str:
        """Return the selected dataset-local channel identifier."""
        return str(self.payload["channel_id"])


class RasterToolbarActionEvent(RasterEvent):
    """Report one discrete JavaScript toolbar action."""

    @property
    def action(self) -> str:
        """Return the stable action name."""
        return str(self.payload.get("action", ""))

    @property
    def channel_id(self) -> str | None:
        """Return the selected channel ID when the action carries one."""
        value = self.payload.get("channel_id")
        return str(value) if value is not None else None


class RasterPlaneChangeEvent(RasterEvent):
    """Report a committed T/Z plane or sliding-Z projection change."""

    @property
    def t_index(self) -> int | None:
        """Return the selected zero-based T index when present."""
        value = self.payload.get("t_index")
        return int(value) if value is not None else None

    @property
    def z_index(self) -> int | None:
        """Return the selected zero-based Z index when present."""
        value = self.payload.get("z_index")
        return int(value) if value is not None else None

    @property
    def plus_minus_z(self) -> int:
        """Return the non-negative sliding-Z projection radius."""
        return int(self.payload.get("plus_minus_z", 0))


class RasterPerformanceEvent(RasterEvent):
    """Report one browser fetch or rendering timing measurement."""

    @property
    def phase(self) -> str:
        """Return the measured phase name, such as plane-fetch or plane-update."""
        return str(self.payload.get("phase", ""))


class RasterRoiSelectedEvent(RasterEvent):
    """Report user-originated ROI selection."""

    @property
    def roi_id(self) -> int | None:
        """Return the selected ROI ID, or None when selection was cleared."""
        value = self.payload.get("roi_id")
        return int(value) if value is not None else None


class RasterRoiCreateRequestedEvent(RasterEvent):
    """Report an uncommitted typed ROI-creation proposal."""

    @property
    def specification(self) -> RoiCreate:
        """Return proposed geometry and metadata in source coordinates."""
        roi_type = RoiType(str(self.payload.get("roi_type")))
        name = str(self.payload.get("name", ""))
        note = str(self.payload.get("note", ""))
        data = self.payload.get("data")
        if roi_type is RoiType.RECTROI:
            return RectRoiCreate(name, RectRoiBounds.from_mapping(data), note)
        return LineRoiCreate(name, LineEndpoints.from_mapping(data), note)


class RasterRoiEditCommittedEvent(RasterEvent):
    """Report an uncommitted typed ROI-edit proposal."""

    @property
    def roi(self) -> Roi:
        """Return the proposed complete ROI for Python validation."""
        return roi_from_mapping(self.payload.get("roi"))


class RasterRoiStateChangeEvent(RasterEvent):
    """Report transition among idle, creating, and editing ROI states."""

    @property
    def state(self) -> RoiInteractionState:
        """Return the validated interaction-state name."""
        return RoiInteractionState(str(self.payload.get("state", "")))
