"""Event overlay sub-API for ``EChartWidget``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from nicewidgets.echart_widget.models import EChartEventOverlay


class _WidgetWithApply(Protocol):
    """Protocol for the owning widget used by the overlay API."""

    def apply(self) -> None:
        """Apply current widget state to the browser chart."""


@dataclass(frozen=True, slots=True)
class EventStyle:
    """Rendering style for one event type.

    Args:
        fill_color: MarkArea fill color.
        line_color: Border color.
        line_width: Border width in pixels.
        line_style: Border style such as ``solid``, ``dashed``, or ``dotted``.
    """

    fill_color: str
    line_color: str
    line_width: int = 2
    line_style: str = "solid"


EVENT_STYLE_BY_TYPE: dict[str, EventStyle] = {
    "user": EventStyle(fill_color="rgba(30, 144, 255, 0.18)", line_color="#1e90ff", line_width=2),
    "rise": EventStyle(fill_color="rgba(46, 204, 113, 0.18)", line_color="#2ecc71", line_width=2),
    "fall": EventStyle(fill_color="rgba(231, 76, 60, 0.18)", line_color="#e74c3c", line_width=2),
    "transient": EventStyle(fill_color="rgba(155, 89, 182, 0.18)", line_color="#9b59b6", line_width=2),
}
SELECTED_EVENT_STYLE = EventStyle(
    fill_color="rgba(255, 221, 0, 0.30)",
    line_color="#ffdd00",
    line_width=4,
    line_style="solid",
)


class EChartEventOverlayApi:
    """Logical sub-API for x-span event overlays on an ``EChartWidget``."""

    def __init__(self, owner: _WidgetWithApply) -> None:
        """Create the overlay API.

        Args:
            owner: Owning widget that provides ``apply``.
        """
        self._owner = owner
        self._events: dict[str, EChartEventOverlay] = {}
        self._selected_event_id: str | None = None
        self.visible = True

    @property
    def selected_event_id(self) -> str | None:
        """Return the selected event id, if any."""
        return self._selected_event_id

    def set_events(self, events: Sequence[object]) -> None:
        """Replace all overlays.

        Args:
            events: Sequence of ``EChartEventOverlay`` or compatible objects.
        """
        overlays = [self._coerce_event(event) for event in events]
        self._events = {overlay.id: overlay for overlay in overlays}
        if self._selected_event_id not in self._events:
            self._selected_event_id = None
        self._owner.apply()

    def add_event(self, event: object) -> EChartEventOverlay:
        """Add or replace one overlay.

        Args:
            event: Event overlay or compatible object.

        Returns:
            Stored overlay.
        """
        overlay = self._coerce_event(event)
        self._events[overlay.id] = overlay
        self._owner.apply()
        return overlay

    def delete_event(self, event_id: str | int) -> None:
        """Delete one overlay if present.

        Args:
            event_id: Event id.
        """
        sid = str(event_id)
        self._events.pop(sid, None)
        if self._selected_event_id == sid:
            self._selected_event_id = None
        self._owner.apply()

    def update_event(self, event: object) -> EChartEventOverlay:
        """Update one overlay.

        Args:
            event: Replacement event overlay.

        Returns:
            Stored overlay.
        """
        overlay = self._coerce_event(event)
        if overlay.id not in self._events:
            raise KeyError(f"event id not found: {overlay.id}")
        self._events[overlay.id] = overlay
        self._owner.apply()
        return overlay

    def select_event(self, event_id: str | int | None) -> None:
        """Set selected overlay id.

        Args:
            event_id: Event id, or None to clear selection.
        """
        selected = None if event_id is None else str(event_id)
        if selected is not None and selected not in self._events:
            raise KeyError(f"event id not found: {selected}")
        self._selected_event_id = selected
        self._owner.apply()

    def clear_events(self) -> None:
        """Clear all overlays and selection."""
        self._events.clear()
        self._selected_event_id = None
        self._owner.apply()

    def set_visible(self, visible: bool) -> None:
        """Show or hide all overlays.

        Args:
            visible: Desired visibility.
        """
        self.visible = bool(visible)
        self._owner.apply()

    def get_events(self) -> list[EChartEventOverlay]:
        """Return overlays sorted by id string."""
        return [self._events[key] for key in sorted(self._events)]

    def build_mark_area(self) -> dict[str, object]:
        """Build ECharts ``markArea`` options for current overlays.

        The method always returns a ``markArea`` dictionary, even when overlays
        are hidden or empty. ECharts merges option updates by default, so
        omitting ``markArea`` does not reliably clear previously drawn
        rectangles. Returning an explicit empty ``data`` list lets the owning
        widget clear stale event rectangles when visibility is turned off.

        Returns:
            MarkArea options with zero or more data entries.
        """
        if not self.visible or not self._events:
            return {"silent": False, "data": []}
        return {
            "silent": False,
            "data": [self._event_to_mark_area(event) for event in self.get_events()],
        }

    def _event_to_mark_area(self, event: EChartEventOverlay) -> list[dict[str, object]]:
        """Convert one overlay to ECharts markArea data."""
        style = self._style_for(event)
        x0, x1 = sorted((float(event.x0), float(event.x1)))
        return [
            {
                "name": event.id,
                "xAxis": x0,
                "itemStyle": {
                    "color": style.fill_color,
                    "borderColor": style.line_color,
                    "borderWidth": style.line_width,
                    "borderType": style.line_style,
                },
                "label": {"show": False},
            },
            {"xAxis": x1},
        ]

    def _style_for(self, event: EChartEventOverlay) -> EventStyle:
        """Return resolved GUI style for an overlay."""
        if event.id == self._selected_event_id:
            return SELECTED_EVENT_STYLE
        return EVENT_STYLE_BY_TYPE.get(event.event_type, EVENT_STYLE_BY_TYPE["user"])

    @staticmethod
    def _coerce_event(event: object) -> EChartEventOverlay:
        """Coerce supported event-like objects to ``EChartEventOverlay``."""
        if isinstance(event, EChartEventOverlay):
            return event
        return EChartEventOverlay.from_object(event)
