"""Interactive demo for the NiceWidgets PlotlyPlotWidget.

Run with:

    uv run python scripts/nicewidgets/try_plotly_plot_widget.py

The demo exercises the public widget API that CloudScope views will eventually
use: continuous traces, sparse scatter overlays, programmatic x-axis limits,
x-range callbacks from user zoom/pan, a draggable horizontal line, and a
vertical measurement pair.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from nicegui import ui

from nicewidgets.plotly_plot.models import MeasurementChangeEvent
from nicewidgets.plotly_plot.widget import PlotlyPlotWidget


def _time_values(n: int = 40_000) -> list[float]:
    """Return synthetic time values for the demo.

    Args:
        n: Number of sample points.

    Returns:
        Monotonic time values in seconds.
    """
    return [i * 0.001 for i in range(n)]


def _signal_values(x: Sequence[float]) -> list[float]:
    """Return a synthetic fluorescence-like signal.

    Args:
        x: Time values in seconds.

    Returns:
        Synthetic y-values containing oscillations and sparse peaks.
    """
    values: list[float] = []
    for t in x:
        baseline = 1.0 + 0.04 * math.sin(t * 0.7)
        carrier = 0.08 * math.sin(t * 8.0)
        peak = 0.0
        for center in (4.0, 11.0, 17.5, 29.0, 35.0):
            peak += 0.45 * math.exp(-((t - center) ** 2) / 0.015)
        values.append(baseline + carrier + peak)
    return values


def _peak_points(x: Sequence[float], y: Sequence[float]) -> tuple[list[float], list[float]]:
    """Return sparse peak marker coordinates for the demo.

    Args:
        x: Time values in seconds.
        y: Signal values.

    Returns:
        Pair of x/y marker lists.
    """
    centers = [4.0, 11.0, 17.5, 29.0, 35.0]
    xs: list[float] = []
    ys: list[float] = []
    for center in centers:
        index = min(range(len(x)), key=lambda i: abs(x[i] - center))
        xs.append(float(x[index]))
        ys.append(float(y[index]))
    return xs, ys


def main() -> None:
    """Build and run the PlotlyPlotWidget demo app."""
    ui.label("NiceWidgets PlotlyPlotWidget demo").classes("text-h5")
    ui.label(
        "Drag the horizontal threshold line, drag either vertical interval line, "
        "or click+drag the plot to zoom the x-axis."
    )

    x = _time_values()
    y = _signal_values(x)
    peak_x, peak_y = _peak_points(x, y)

    x_range_label = ui.label("x-range: auto")
    measurement_label = ui.label("measurement: none")

    def on_x_range_changed(x_min: float | None, x_max: float | None) -> None:
        """Report user x-axis range changes.

        Args:
            x_min: New x-axis minimum, or ``None`` for autorange.
            x_max: New x-axis maximum, or ``None`` for autorange.
        """
        if x_min is None or x_max is None:
            x_range_label.text = "x-range: auto"
            return
        x_range_label.text = f"x-range: {x_min:.3f} to {x_max:.3f} s"

    def on_measurement_changed(event: MeasurementChangeEvent) -> None:
        """Report user measurement-line drags.

        Args:
            event: Measurement callback payload from the widget.
        """
        if event.kind == "line":
            measurement_label.text = f"{event.name}: {event.position:.4f}"
            return
        measurement_label.text = (
            f"{event.name}: {event.position1:.3f} to {event.position2:.3f}; "
            f"delta={event.delta:.3f}"
        )

    plot = PlotlyPlotWidget(
        x_label="Time (s)",
        y_label="Normalized intensity",
        on_x_range_changed=on_x_range_changed,
        on_measurement_changed=on_measurement_changed,
    )
    plot.container.classes("w-full h-96")
    plot.add_trace(name="normalized intensity", x=x, y=y)
    plot.plot_scatter(name="peaks", x=peak_x, y=peak_y)
    plot.add_measurement_line(name="manual threshold", orientation="horizontal", value=1.25)
    plot.add_measurement_pair(name="time interval", orientation="vertical", value1=8.0, value2=12.0)

    with ui.row():
        ui.button("Zoom 8-14 s", on_click=lambda: plot.set_x_axis_limits(8.0, 14.0))
        ui.button("Reset x-axis", on_click=plot.reset_x_axis_limits)
        ui.button(
            "Move peaks",
            on_click=lambda: plot.update_scatter(
                name="peaks",
                x=[value + 0.25 for value in peak_x],
                y=peak_y,
            ),
        )

    ui.run(title="PlotlyPlotWidget demo", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
