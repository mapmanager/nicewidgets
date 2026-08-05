"""Typed public models for non-interactive X/Y plot overlays."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class XYPlotMode(StrEnum):
    """Supported X/Y plot drawing modes."""

    MARKERS = "markers"
    LINES = "lines"
    LINES_MARKERS = "lines_markers"


@dataclass(frozen=True, slots=True)
class XYPlotStyle:
    """Visual styling applied to one X/Y plot.

    Attributes:
        color: Any browser-supported CSS color.
        marker_size: Marker diameter in screen pixels.
        line_width: Line width in screen pixels.
        opacity: Plot opacity from zero through one.
    """

    color: str = "#facc15"
    marker_size: float = 5.0
    line_width: float = 1.5
    opacity: float = 1.0

    def __post_init__(self) -> None:
        """Validate style values at the Python API boundary."""
        if not self.color.strip():
            raise ValueError("color must not be empty")
        if not math.isfinite(self.marker_size) or self.marker_size <= 0:
            raise ValueError("marker_size must be finite and greater than zero")
        if not math.isfinite(self.line_width) or self.line_width <= 0:
            raise ValueError("line_width must be finite and greater than zero")
        if not math.isfinite(self.opacity) or not 0 <= self.opacity <= 1:
            raise ValueError("opacity must be finite and between zero and one")

    def to_json(self) -> dict[str, str | float]:
        """Return a JSON-compatible snake_case style mapping."""
        return {
            "color": self.color,
            "marker_size": self.marker_size,
            "line_width": self.line_width,
            "opacity": self.opacity,
        }


def _coordinates(values: Iterable[float | None]) -> tuple[float | None, ...]:
    """Normalize non-finite coordinates to JSON-compatible gaps."""
    normalized: list[float | None] = []
    for value in values:
        if value is None:
            normalized.append(None)
            continue
        number = float(value)
        normalized.append(number if math.isfinite(number) else None)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class XYPlot:
    """One addressable, non-interactive X/Y plot overlay.

    Coordinates are expressed in the viewer's physical display coordinate
    space. Values outside the image are retained and can become visible after
    panning or zooming. A non-finite X or Y value creates a gap: its marker is
    omitted and line segments do not cross it.

    ``point_ids`` are optional stable identities reserved for future point
    interaction. ``z_index`` currently filters the whole plot to one Z plane;
    the preserved point indexing makes a future per-point Z/T extension
    straightforward.
    """

    plot_id: str
    x: Iterable[float | None]
    y: Iterable[float | None]
    name: str | None = None
    mode: XYPlotMode = XYPlotMode.MARKERS
    style: XYPlotStyle = field(default_factory=XYPlotStyle)
    visible: bool = True
    channel_ids: Sequence[str] | None = None
    z_index: int | None = None
    point_ids: Sequence[str] | None = None
    coordinate_space: str = "physical"

    def __post_init__(self) -> None:
        """Normalize coordinates and validate the complete plot contract."""
        plot_id = self.plot_id.strip()
        if not plot_id:
            raise ValueError("plot_id must not be empty")
        x = _coordinates(self.x)
        y = _coordinates(self.y)
        if len(x) != len(y):
            raise ValueError("x and y must have equal lengths")
        if self.coordinate_space != "physical":
            raise ValueError("coordinate_space must be 'physical'")
        if self.z_index is not None and self.z_index < 0:
            raise ValueError("z_index must be non-negative")
        channel_ids = None
        if self.channel_ids is not None:
            channel_ids = tuple(value.strip() for value in self.channel_ids)
            if any(not value for value in channel_ids):
                raise ValueError("channel_ids must not contain empty values")
            if len(set(channel_ids)) != len(channel_ids):
                raise ValueError("channel_ids must be unique")
        point_ids = None
        if self.point_ids is not None:
            point_ids = tuple(value.strip() for value in self.point_ids)
            if len(point_ids) != len(x):
                raise ValueError("point_ids must have the same length as x and y")
            if any(not value for value in point_ids):
                raise ValueError("point_ids must not contain empty values")
            if len(set(point_ids)) != len(point_ids):
                raise ValueError("point_ids must be unique")
        object.__setattr__(self, "plot_id", plot_id)
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "channel_ids", channel_ids)
        object.__setattr__(self, "point_ids", point_ids)

    def to_json(self) -> dict[str, object]:
        """Return the exact snake_case JavaScript plot envelope."""
        return {
            "plot_id": self.plot_id,
            "name": self.name,
            "x": list(self.x),
            "y": list(self.y),
            "mode": self.mode.value,
            "style": self.style.to_json(),
            "visible": self.visible,
            "channel_ids": list(self.channel_ids) if self.channel_ids is not None else None,
            "z_index": self.z_index,
            "point_ids": list(self.point_ids) if self.point_ids is not None else None,
            "coordinate_space": self.coordinate_space,
        }
