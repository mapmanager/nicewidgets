"""Event overlay sub-API for ``PlotlyPlotWidget``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from nicewidgets.echart_widget.event_overlay import (
    EVENT_STYLE_BY_TYPE,
    SELECTED_EVENT_STYLE,
    EventStyle,
)


@dataclass(frozen=True, slots=True)
class PlotlyEventOverlay:
    """GUI-facing x-span event overlay as Plotly layout ``rect`` shapes.

    Args:
        id: Stable event id as a string.
        x0: First x coordinate.
        x1: Second x coordinate.
        event_type: Event type key used for GUI style lookup.
    """

    id: str
    x0: float
    x1: float
    event_type: str = "user"

    @classmethod
    def from_object(cls, obj: object) -> PlotlyEventOverlay:
        """Adapt a dataclass-like event object into an overlay.

        Args:
            obj: Object with ``id``, ``x0``, ``x1``, and optional ``event_type``.

        Returns:
            Event overlay instance.
        """
        event_type = getattr(obj, "event_type", "user")
        if hasattr(event_type, "value"):
            event_type = event_type.value
        return cls(
            id=str(obj.id),
            x0=float(obj.x0),
            x1=float(obj.x1),
            event_type=str(event_type),
        )


class _WidgetWithEventApply(Protocol):
    """Protocol for the owning widget used by the overlay API."""

    def _apply_event_overlays(self) -> None:
        """Push merged measurement + event shapes to the browser."""


class PlotlyEventOverlayApi:
    """Logical sub-API for x-span event overlays on a ``PlotlyPlotWidget``."""

    def __init__(self, owner: _WidgetWithEventApply) -> None:
        """Create the overlay API.

        Args:
            owner: Owning widget that applies layout shapes.
        """
        self._owner = owner
        self._events: dict[str, PlotlyEventOverlay] = {}
        self._selected_event_id: str | None = None
        self.visible = True

    @property
    def selected_event_id(self) -> str | None:
        """Return the selected event id, if any."""
        return self._selected_event_id

    def set_events(self, events: Sequence[object]) -> None:
        """Replace all overlays.

        Args:
            events: Sequence of ``PlotlyEventOverlay`` or compatible objects.
        """
        overlays = [self._coerce_event(event) for event in events]
        self._events = {overlay.id: overlay for overlay in overlays}
        if self._selected_event_id not in self._events:
            self._selected_event_id = None
        self._owner._apply_event_overlays()

    def add_event(self, event: object) -> PlotlyEventOverlay:
        """Add or replace one overlay."""
        overlay = self._coerce_event(event)
        self._events[overlay.id] = overlay
        self._owner._apply_event_overlays()
        return overlay

    def delete_event(self, event_id: str | int) -> None:
        """Delete one overlay if present."""
        sid = str(event_id)
        self._events.pop(sid, None)
        if self._selected_event_id == sid:
            self._selected_event_id = None
        self._owner._apply_event_overlays()

    def update_event(self, event: object) -> PlotlyEventOverlay:
        """Update one overlay."""
        overlay = self._coerce_event(event)
        if overlay.id not in self._events:
            raise KeyError(f"event id not found: {overlay.id}")
        self._events[overlay.id] = overlay
        self._owner._apply_event_overlays()
        return overlay

    def select_event(self, event_id: str | int | None) -> None:
        """Set selected overlay id."""
        selected = None if event_id is None else str(event_id)
        if selected is not None and selected not in self._events:
            raise KeyError(f"event id not found: {selected}")
        self._selected_event_id = selected
        self._owner._apply_event_overlays()

    def clear_events(self) -> None:
        """Clear all overlays and selection."""
        self._events.clear()
        self._selected_event_id = None
        self._owner._apply_event_overlays()

    def set_visible(self, visible: bool) -> None:
        """Show or hide all overlays."""
        self.visible = bool(visible)
        self._owner._apply_event_overlays()

    def get_events(self) -> list[PlotlyEventOverlay]:
        """Return overlays sorted by id string."""
        return [self._events[key] for key in sorted(self._events)]

    def build_plotly_shapes(self) -> list[dict[str, Any]]:
        """Build non-editable Plotly ``rect`` shapes for current overlays."""
        if not self.visible or not self._events:
            return []
        return [self._event_to_shape(event) for event in self.get_events()]

    def _event_to_shape(self, event: PlotlyEventOverlay) -> dict[str, Any]:
        """Convert one overlay to a Plotly layout shape dict."""
        style = self._style_for(event)
        x0, x1 = sorted((float(event.x0), float(event.x1)))
        line: dict[str, Any] = {
            "color": style.line_color,
            "width": style.line_width,
        }
        if style.line_style != "solid":
            line["dash"] = style.line_style
        return {
            "type": "rect",
            "name": f"event:{event.id}",
            "x0": x0,
            "x1": x1,
            "y0": 0,
            "y1": 1,
            "xref": "x",
            "yref": "paper",
            "fillcolor": style.fill_color,
            "line": line,
            "editable": False,
            "layer": "below",
        }

    def _style_for(self, event: PlotlyEventOverlay) -> EventStyle:
        """Return resolved GUI style for an overlay."""
        if event.id == self._selected_event_id:
            return SELECTED_EVENT_STYLE
        return EVENT_STYLE_BY_TYPE.get(event.event_type, EVENT_STYLE_BY_TYPE["user"])

    @staticmethod
    def _coerce_event(event: object) -> PlotlyEventOverlay:
        """Coerce supported event-like objects to ``PlotlyEventOverlay``."""
        if isinstance(event, PlotlyEventOverlay):
            return event
        return PlotlyEventOverlay.from_object(event)
