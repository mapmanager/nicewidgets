"""Data models for the NiceWidgets ECharts line-plot widget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EChartLineData:
    """One line series for an ECharts value/value plot.

    Args:
        x: X-axis values in data coordinates.
        y: Y-axis values in data coordinates.
        x_label: Human-readable x-axis label.
        y_label: Human-readable y-axis label.
        series_name: Human-readable series name.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    x_label: str
    y_label: str
    series_name: str = "series"

    @classmethod
    def from_sequences(
        cls,
        *,
        x: Sequence[float],
        y: Sequence[float],
        x_label: str,
        y_label: str,
        series_name: str = "series",
    ) -> EChartLineData:
        """Create line data from numeric sequences.

        Args:
            x: X-axis values.
            y: Y-axis values.
            x_label: Human-readable x-axis label.
            y_label: Human-readable y-axis label.
            series_name: Human-readable series name.

        Returns:
            Immutable line-data object.

        Raises:
            ValueError: If x and y lengths differ.
        """
        x_values = tuple(float(value) for value in x)
        y_values = tuple(float(value) for value in y)
        if len(x_values) != len(y_values):
            raise ValueError(
                f"x and y must have the same length, got {len(x_values)} and {len(y_values)}"
            )
        return cls(
            x=x_values,
            y=y_values,
            x_label=str(x_label),
            y_label=str(y_label),
            series_name=str(series_name),
        )


@dataclass(frozen=True, slots=True)
class EChartAxisRange:
    """Optional x-axis range for an ECharts value axis.

    Args:
        x_min: Minimum x-axis value, or None for auto.
        x_max: Maximum x-axis value, or None for auto.
    """

    x_min: float | None = None
    x_max: float | None = None

    def __post_init__(self) -> None:
        """Validate axis range.

        Raises:
            ValueError: If both bounds are set and x_min >= x_max.
        """
        if self.x_min is not None and self.x_max is not None and self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")


@dataclass(frozen=True, slots=True)
class EChartEventOverlay:
    """GUI-facing x-span event overlay for an ECharts markArea.

    This model intentionally does not import or depend on AcqStore. Callers can
    adapt backend event objects into this lightweight shape.

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
    def from_object(cls, obj: object) -> EChartEventOverlay:
        """Adapt a dataclass-like event object into an overlay.

        Args:
            obj: Object with ``id``, ``x0``, ``x1``, and optional
                ``event_type`` attributes.

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
