"""Interactive demo for PlotlyPlotWidget event overlays + box-select (Phase 2 gate).

Run with:

    uv run python scripts/nicewidgets/try_plotly_plot_event_overlays.py

Models ``AcqAnalysisPlotView``: velocity trace, event x-span rects, arm/cancel
box-select, and ``on_x_range_selected`` → add event. Relayout args log at INFO.
"""

from __future__ import annotations

import math

from nicegui import ui

from nicewidgets.plotly_plot.event_overlay import PlotlyEventOverlay
from nicewidgets.plotly_plot.widget import PlotlyPlotWidget


def _velocity_like_trace(n: int = 5_000) -> tuple[list[float], list[float]]:
    """Return synthetic velocity-like x/y for a single-subplot line plot."""
    x = [i * 0.001 for i in range(n)]
    y = [0.5 * math.sin(t * 3.0) + 0.1 * math.sin(t * 17.0) for t in x]
    return x, y


def main() -> None:
    """Build and run the acq-analysis-like demo."""
    ui.label("PlotlyPlotWidget — acq analysis plot demo").classes("text-h5")
    ui.label(
        "Arm box-select and drag to add a user event (like Add Event in CloudScope). "
        "Relayout args log to the terminal at INFO."
    ).classes("text-sm opacity-80")

    x, y = _velocity_like_trace()
    next_event_id = {"value": 1}

    x_range_label = ui.label("x-range: auto").classes("font-mono text-sm")
    selected_label = ui.label("box-select: (none)").classes("font-mono text-sm")
    events_label = ui.label("events: 0").classes("font-mono text-sm")

    def refresh_events_label() -> None:
        events = plot.events.get_events()
        events_label.text = f"events: {len(events)} — " + ", ".join(
            f"{e.id}[{e.event_type}]" for e in events
        ) or "(none)"

    def on_x_range_changed(x_min: float | None, x_max: float | None) -> None:
        if x_min is None or x_max is None:
            x_range_label.text = "x-range: auto"
            print("on_x_range_changed: auto")
            return
        msg = f"x-range: {x_min:.4f} to {x_max:.4f} s"
        x_range_label.text = msg
        print(f"on_x_range_changed: {msg}")

    def on_x_range_selected(x0: float, x1: float) -> None:
        eid = str(next_event_id["value"])
        next_event_id["value"] += 1
        plot.events.add_event(PlotlyEventOverlay(id=eid, x0=x0, x1=x1, event_type="user"))
        plot.events.select_event(eid)
        selected_label.text = f"box-select: added event {eid} ({min(x0, x1):.4f}–{max(x0, x1):.4f} s)"
        print(f"on_x_range_selected: added event {eid}")
        refresh_events_label()

    plot = PlotlyPlotWidget(
        x_label="Time (s)",
        y_label="Velocity",
        on_x_range_changed=on_x_range_changed,
        on_x_range_selected=on_x_range_selected,
    )
    plot.container.classes("w-full h-96")
    plot.add_trace(name="velocity", x=x, y=y)

    plot.events.set_events(
        [
            PlotlyEventOverlay(id="seed-rise", x0=1.0, x1=1.4, event_type="rise"),
            PlotlyEventOverlay(id="seed-fall", x0=2.2, x1=2.7, event_type="fall"),
        ]
    )
    refresh_events_label()

    def arm_selection() -> None:
        plot.begin_select_x_range()
        print("begin_select_x_range()")

    def cancel_selection() -> None:
        plot.cancel_select_x_range()
        print("cancel_select_x_range()")

    with ui.row().classes("gap-2 flex-wrap"):
        ui.button("Arm box-select (add event)", on_click=arm_selection)
        ui.button("Cancel box-select", on_click=cancel_selection)
        ui.button("Select seed-rise", on_click=lambda: plot.events.select_event("seed-rise"))
        ui.button("Hide events", on_click=lambda: (plot.events.set_visible(False), refresh_events_label()))
        ui.button("Show events", on_click=lambda: (plot.events.set_visible(True), refresh_events_label()))
        ui.button("Clear events", on_click=lambda: (plot.events.clear_events(), refresh_events_label()))
        ui.button("Zoom 2–8 s", on_click=lambda: plot.set_x_axis_limits(2.0, 8.0))
        ui.button("Reset x-axis", on_click=plot.reset_x_axis_limits)

    ui.run(title="PlotlyPlotWidget event overlays demo", reload=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
