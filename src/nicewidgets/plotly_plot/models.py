"""Public models for the NiceWidgets Plotly plot widget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

PlotlyLineOrientation = Literal["horizontal", "vertical"]
PlotlyMeasurementKind = Literal["line", "pair"]
PlotlySeriesKind = Literal["trace", "scatter"]
PlotlyYAxisSide = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class PlotlySeriesMenuItem:
    """One trace or scatter overlay entry in the Plotly plot context menu.

    Args:
        series_name: Stable series name matching ``PlotlyTraceData.name`` or
            ``PlotlyScatterData.name``.
        label: Human-readable menu label.
        default_visible: Initial visibility when the series is first registered.
        kind: Whether the series is a continuous trace or scatter overlay.
        separator_before: When True, render a menu separator immediately before
            this item in the plot context menu.
    """

    series_name: str
    label: str
    default_visible: bool = True
    kind: PlotlySeriesKind = "trace"
    separator_before: bool = False


def _as_float_tuple(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    """Convert a numeric sequence to a tuple of floats.

    Args:
        values: Numeric values to convert.
        name: Human-readable value name for validation errors.

    Returns:
        Tuple containing float-converted values.

    Raises:
        ValueError: If the sequence is empty.
    """
    result = tuple(float(value) for value in values)
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


@dataclass(frozen=True, slots=True)
class PlotlyTraceData:
    """Continuous x/y trace data for a Plotly ``scattergl`` trace.

    Args:
        name: Stable caller-defined trace name.
        x: X-axis values in data coordinates.
        y: Y-axis values in data coordinates.
        visible: Whether the trace should be visible.
        y_axis: Primary ``y`` axis (``"left"``) or overlaid ``y2`` axis (``"right"``).
        line_color: Optional Plotly line color.
        line_dash: Optional Plotly line dash (for example ``"solid"``, ``"dot"``).
    """

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    visible: bool = True
    y_axis: PlotlyYAxisSide = "left"
    line_color: str | None = None
    line_dash: str | None = None

    @classmethod
    def from_sequences(
        cls,
        *,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        visible: bool = True,
        y_axis: PlotlyYAxisSide = "left",
        line_color: str | None = None,
        line_dash: str | None = None,
    ) -> PlotlyTraceData:
        """Create trace data from numeric sequences.

        Args:
            name: Stable caller-defined trace name.
            x: X-axis values in data coordinates.
            y: Y-axis values in data coordinates.
            visible: Whether the trace should be visible.
            y_axis: Primary ``y`` axis (``"left"``) or overlaid ``y2`` axis (``"right"``).
            line_color: Optional Plotly line color.
            line_dash: Optional Plotly line dash.

        Returns:
            Immutable trace data.

        Raises:
            ValueError: If ``name`` is empty, ``x`` or ``y`` is empty, or lengths differ.
        """
        trace_name = str(name).strip()
        if not trace_name:
            raise ValueError("trace name must not be empty")
        x_values = _as_float_tuple(x, name="x")
        y_values = _as_float_tuple(y, name="y")
        if len(x_values) != len(y_values):
            raise ValueError(
                f"x and y must have the same length, got {len(x_values)} and {len(y_values)}"
            )
        axis = _normalize_y_axis_side(y_axis)
        return cls(
            name=trace_name,
            x=x_values,
            y=y_values,
            visible=bool(visible),
            y_axis=axis,
            line_color=None if line_color is None else str(line_color),
            line_dash=None if line_dash is None else str(line_dash),
        )


@dataclass(frozen=True, slots=True)
class PlotlyScatterData:
    """Sparse x/y marker overlay for a Plotly ``scattergl`` trace.

    Args:
        name: Stable caller-defined scatter overlay name.
        x: X-axis values in data coordinates.
        y: Y-axis values in data coordinates.
        visible: Whether the scatter overlay should be visible.
        y_axis: Primary ``y`` axis (``"left"``) or overlaid ``y2`` axis (``"right"``).
    """

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    visible: bool = True
    y_axis: PlotlyYAxisSide = "left"

    @classmethod
    def from_sequences(
        cls,
        *,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        visible: bool = True,
        y_axis: PlotlyYAxisSide = "left",
    ) -> PlotlyScatterData:
        """Create scatter overlay data from numeric sequences.

        Args:
            name: Stable caller-defined scatter overlay name.
            x: X-axis values in data coordinates.
            y: Y-axis values in data coordinates.
            visible: Whether the scatter overlay should be visible.
            y_axis: Primary ``y`` axis (``"left"``) or overlaid ``y2`` axis (``"right"``).

        Returns:
            Immutable scatter overlay data.

        Raises:
            ValueError: If ``name`` is empty, ``x`` or ``y`` is empty, or lengths differ.
        """
        scatter_name = str(name).strip()
        if not scatter_name:
            raise ValueError("scatter name must not be empty")
        x_values = _as_float_tuple(x, name="x")
        y_values = _as_float_tuple(y, name="y")
        if len(x_values) != len(y_values):
            raise ValueError(
                f"x and y must have the same length, got {len(x_values)} and {len(y_values)}"
            )
        axis = _normalize_y_axis_side(y_axis)
        return cls(
            name=scatter_name,
            x=x_values,
            y=y_values,
            visible=bool(visible),
            y_axis=axis,
        )


def _normalize_y_axis_side(value: str | PlotlyYAxisSide) -> PlotlyYAxisSide:
    """Normalize a y-axis side literal.

    Args:
        value: ``"left"`` or ``"right"``.

    Returns:
        Normalized axis side.

    Raises:
        ValueError: If the value is not supported.
    """
    side = str(value).strip().lower()
    if side == "left":
        return "left"
    if side == "right":
        return "right"
    raise ValueError(f"y_axis must be 'left' or 'right', got {value!r}")


@dataclass(frozen=True, slots=True)
class PlotlyAxisRange:
    """Optional x-axis range for a Plotly value axis.

    Args:
        x_min: Minimum x-axis value, or ``None`` for automatic scaling.
        x_max: Maximum x-axis value, or ``None`` for automatic scaling.
    """

    x_min: float | None = None
    x_max: float | None = None

    def __post_init__(self) -> None:
        """Validate axis bounds.

        Raises:
            ValueError: If both bounds are set and ``x_min >= x_max``.
        """
        if (self.x_min is None) != (self.x_max is None):
            raise ValueError("x_min and x_max must either both be set or both be None")
        if self.x_min is not None and self.x_max is not None and self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")


@dataclass(slots=True)
class MeasurementLine:
    """One horizontal or vertical measurement line.

    Args:
        name: Stable caller-defined measurement name.
        orientation: ``"horizontal"`` for a y-value line or ``"vertical"`` for an x-value line.
        position: Current line position in data coordinates.
        visible: Whether the line is visible.
        y_axis: Y-axis for horizontal lines. Ignored for vertical lines.
        editable: Whether the user can drag the line in the browser.
        color: Plotly line color. ``None`` uses a theme-aware default.
        dash: Plotly dash style (for example ``"solid"``, ``"dot"``, ``"dash"``).
        show_legend: Whether the line appears in the Plotly legend.
        legend_label: Legend text when ``show_legend`` is True. Defaults to ``name``.
    """

    name: str
    orientation: PlotlyLineOrientation
    position: float
    visible: bool = True
    y_axis: PlotlyYAxisSide = "left"
    editable: bool = True
    color: str | None = None
    dash: str = "dash"
    show_legend: bool = False
    legend_label: str | None = None


@dataclass(slots=True)
class MeasurementPair:
    """Two independently draggable measurement lines with a reported interval.

    Args:
        name: Stable caller-defined measurement-pair name.
        orientation: ``"horizontal"`` for y-value lines or ``"vertical"`` for x-value lines.
        position1: Current first-line position in data coordinates.
        position2: Current second-line position in data coordinates.
        visible: Whether both lines are visible.
        y_axis: Y-axis for horizontal lines. Ignored for vertical lines.
    """

    name: str
    orientation: PlotlyLineOrientation
    position1: float
    position2: float
    visible: bool = True
    y_axis: PlotlyYAxisSide = "left"

    @property
    def delta(self) -> float:
        """Return the absolute distance between both measurement positions."""
        return abs(self.position2 - self.position1)


@dataclass(frozen=True, slots=True)
class MeasurementChangeEvent:
    """Measurement callback payload emitted after a user drags a line.

    Args:
        name: Stable caller-defined measurement name.
        kind: ``"line"`` for a single line or ``"pair"`` for a linked pair.
        orientation: Measurement orientation.
        position: Single-line position, or the moved line position for a pair.
        position1: First pair position. ``None`` for single-line measurements.
        position2: Second pair position. ``None`` for single-line measurements.
        delta: Absolute pair distance. ``None`` for single-line measurements.
        y_axis: Y-axis scale for horizontal measurements. Always ``"left"`` for vertical.
    """

    name: str
    kind: PlotlyMeasurementKind
    orientation: PlotlyLineOrientation
    position: float
    position1: float | None = None
    position2: float | None = None
    delta: float | None = None
    y_axis: PlotlyYAxisSide = "left"
