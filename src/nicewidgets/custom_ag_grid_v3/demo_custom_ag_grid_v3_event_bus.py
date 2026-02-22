# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/demo_custom_ag_grid_v3_event_bus.py
#
# Step 3 demo: Event-bus integration simulator for CustomAgGrid_v3.
#
# Goals:
# - Grid is a CONSUMER of external events (SelectRow / ReplaceRows / ClearSelection)
# - Exercise "bad order" calls (select before grid ready, select after set_data, etc.)
# - Visible event log panel + simple buttons to generate events
# - Keep grid instance stable (no recreation), use only public API
#
# Run:
#   uv run python ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/demo_custom_ag_grid_v3_event_bus.py

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from nicegui import ui

from nicewidgets.custom_ag_grid_v3.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid_v3.custom_ag_grid_v3 import CustomAgGrid_v3, CellEditEvent
from nicewidgets.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

Row = Dict[str, Any]
Rows = List[Row]
RowId = Union[int, str]


# ---------------------------
# Event bus (tiny, explicit)
# ---------------------------

@dataclass(frozen=True)
class SelectRowEvent:
    row_id: RowId
    ensure_visible: bool = True


@dataclass(frozen=True)
class ReplaceRowsEvent:
    rows: Rows
    reason: str = "replace_rows"


@dataclass(frozen=True)
class ClearSelectionEvent:
    pass


BusEvent = Union[SelectRowEvent, ReplaceRowsEvent, ClearSelectionEvent]


class EventBus:
    """Minimal in-process event bus for the demo."""

    def __init__(self) -> None:
        self._subs: list[Callable[[BusEvent], None]] = []

    def subscribe(self, cb: Callable[[BusEvent], None]) -> None:
        self._subs.append(cb)

    def emit(self, ev: BusEvent) -> None:
        for cb in list(self._subs):
            cb(ev)


# ---------------------------
# Demo data
# ---------------------------

STATUSES = ["new", "reviewed", "bad"]


def make_rows(prefix: str, n: int = 25) -> Rows:
    rows: Rows = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "name": f"{prefix}_{i:02d}",
                "status": STATUSES[i % len(STATUSES)],
                "age": 20 + (i % 7),
                "score": round(50 + (i * 1.7) % 50, 1),
            }
        )
    return rows


def now_s() -> str:
    return time.strftime("%H:%M:%S")


# ---------------------------
# UI
# ---------------------------

_UI_BUILT = False


def build_ui() -> None:
    global _UI_BUILT
    if _UI_BUILT:
        return
    _UI_BUILT = True

    configure_logging(level="DEBUG")

    ui.label("CustomAgGrid_v3 event-bus demo").classes("text-xl font-semibold")
    ui.label("This demo simulates your app: grid consumes events, never recreated.").classes("text-sm opacity-80")
    ui.separator()

    bus = EventBus()

    # Event log (most recent first)
    log_lines: list[str] = []

    def push_log(msg: str) -> None:
        log_lines.insert(0, f"{now_s()}  {msg}")
        # keep small
        del log_lines[50:]
        log_area.value = "\n".join(log_lines)

    # Columns (includes editing so we also stress keynav/edit compatibility)
    columns = [
        ColumnConfig(field="id", header="ID", editable=False),
        ColumnConfig(field="name", header="Name", editable=True, editor="text"),
        ColumnConfig(field="status", header="Status", editable=True, editor="select", choices=STATUSES),
        ColumnConfig(field="age", header="Age", editable=False),
        ColumnConfig(field="score", header="Score", editable=False),
    ]
    cfg = GridConfig(selection_mode="single", stop_editing_on_focus_loss=True)

    selected_label = ui.label("Selected: (none)").classes("text-sm")
    edited_label = ui.label("Last edit: (none)").classes("text-sm")

    # Build grid FIRST so we can test "select before ready" by emitting immediately after construction
    grid = CustomAgGrid_v3(
        make_rows("A"),
        columns,
        cfg,
        row_id_field="id",
        allow_column_reorder=False,
        height_px=320,
        enable_keyboard_row_nav=True,
        suppress_row_hover_highlight=True,
    )
    logger.warning(f"[demo] aggrid theme is: {grid.grid.theme!r}")

    def on_sel(row: Optional[Row]) -> None:
        if not row:
            selected_label.text = "Selected: (none)"
            return
        selected_label.text = (
            f"Selected: id={row.get('id')} name={row.get('name')} status={row.get('status')} "
            f"age={row.get('age')} score={row.get('score')}"
        )

    def on_edit(ev: CellEditEvent) -> None:
        edited_label.text = f"Last edit: row_id={ev.row_id} field={ev.field} {ev.old_value!r} -> {ev.new_value!r}"
        push_log(f"CellEdit row_id={ev.row_id} field={ev.field} {ev.old_value!r}->{ev.new_value!r}")

    grid.on_selection_changed(on_sel)
    grid.on_cell_value_changed(on_edit)

    # Bus consumer
    def handle_event(ev: BusEvent) -> None:
        if isinstance(ev, SelectRowEvent):
            push_log(f"Event SelectRow(row_id={ev.row_id}, ensure_visible={ev.ensure_visible})")
            grid.select_row(ev.row_id, ensure_visible=ev.ensure_visible)
            return
        if isinstance(ev, ReplaceRowsEvent):
            push_log(f"Event ReplaceRows(n={len(ev.rows)}, reason={ev.reason!r})")
            grid.set_data(ev.rows, reason=ev.reason)
            return
        if isinstance(ev, ClearSelectionEvent):
            push_log("Event ClearSelection()")
            grid.clear_selection()
            return

    bus.subscribe(handle_event)

    ui.separator()

    with ui.row().classes("items-center gap-2"):
        ui.button("Emit: SelectRow(20)", on_click=lambda: bus.emit(SelectRowEvent(20, True)))
        ui.button("Emit: SelectRow(3)", on_click=lambda: bus.emit(SelectRowEvent(3, True)))
        ui.button("Emit: ClearSelection", on_click=lambda: bus.emit(ClearSelectionEvent()))

    with ui.row().classes("items-center gap-2"):
        ui.button("Emit: ReplaceRows(A)", on_click=lambda: bus.emit(ReplaceRowsEvent(make_rows("A"), "replace_A")))
        ui.button("Emit: ReplaceRows(B)", on_click=lambda: bus.emit(ReplaceRowsEvent(make_rows("B"), "replace_B")))
        ui.button(
            "Emit: ReplaceRows(B) then SelectRow(12)",
            on_click=lambda: (
                bus.emit(ReplaceRowsEvent(make_rows("B"), "replace_B_then_select")),
                bus.emit(SelectRowEvent(12, True)),
            ),
        )

    with ui.row().classes("items-center gap-2"):
        ui.button(
            "Bad-order test: SelectRow(22) BEFORE ready (run-once)",
            on_click=lambda: bus.emit(SelectRowEvent(22, True)),
        ).props('color="secondary"')
        ui.label("Tip: also resize/sort columns, then ReplaceRows(A/B) and confirm widths/sort persist.").classes(
            "text-xs opacity-80"
        )

    ui.separator()
    ui.label("Event log (most recent first)").classes("text-sm font-semibold")

    log_area = ui.textarea(value="", placeholder="(events will appear here)").props("readonly").classes("w-full")
    log_area.style("height: 180px;")

    # Kick one event immediately after UI build to exercise queued selection.
    push_log("Boot: emitting SelectRow(22) immediately (should queue if grid not ready yet)")
    bus.emit(SelectRowEvent(22, True))


if __name__ in {"__main__", "__mp_main__"}:
    build_ui()
    ui.run(native=True, reload=False, window_size=(1100, 650), port=8001)