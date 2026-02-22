# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/demo_custom_ag_grid_v3_tutorial.py
#
# Tutorial/demo for CustomAgGrid_v3 (v3 widget regression testbed).
#
# Features:
# - Data switching (A/B) via set_data (no grid recreation)
# - Column width + sort state preservation across set_data
# - Row selection callback
# - Keyboard Up/Down prev/next row selection
# - Programmatic row selection + clear selection
# - Double-click editing:
#     * text editor (Name)
#     * select editor (Status)
# - Emits CellEditEvent via on_cell_value_changed
#
# Run:
#   uv run python ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/demo_custom_ag_grid_v3_tutorial.py

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from nicegui import ui

from nicewidgets.custom_ag_grid_v3.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid_v3.custom_ag_grid_v3 import CustomAgGrid_v3, CellEditEvent
from nicewidgets.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

_UI_BUILT = False
STATUSES = ["new", "reviewed", "bad"]


def _make_data_a(n: int = 25) -> List[Dict[str, Any]]:
    return [
        {
            "id": i,
            "name": f"Alice_{i:02d}",
            "age": 20 + (i % 7),
            "score": round(50 + (i * 1.7) % 50, 1),
            "status": STATUSES[i % len(STATUSES)],
        }
        for i in range(n)
    ]


def _make_data_b(n: int = 25) -> List[Dict[str, Any]]:
    return [
        {
            "id": i,
            "name": f"Bob_{i:02d}",
            "age": 30 + (i % 5),
            "score": round(80 - (i * 1.3) % 60, 1),
            "status": STATUSES[(i + 1) % len(STATUSES)],
        }
        for i in range(n)
    ]


def build_ui() -> None:
    global _UI_BUILT
    if _UI_BUILT:
        return
    _UI_BUILT = True

    configure_logging(level="DEBUG")

    ui.label("CustomAgGrid_v3 tutorial").classes("text-xl font-semibold")
    ui.label("Keyboard: click a row, then ArrowUp/ArrowDown to move selection.").classes("text-sm opacity-80")
    ui.label("Editing: double-click Name (text) or Status (select).").classes("text-sm opacity-80")
    ui.separator()

    data_a = _make_data_a()

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

    grid = CustomAgGrid_v3(
        data_a,
        columns,
        cfg,
        row_id_field="id",
        allow_column_reorder=False,
        height_px=320,
        enable_keyboard_row_nav=True,
        suppress_row_hover_highlight=True,
    )

    # abb
    # grid.select_row(20, ensure_visible=True)

    logger.warning(f"[demo] aggrid theme is: {grid.grid.theme!r}")

    def on_sel(row: Optional[Dict[str, Any]]) -> None:
        if not row:
            selected_label.text = "Selected: (none)"
            return
        selected_label.text = (
            f"Selected: id={row.get('id')} name={row.get('name')} status={row.get('status')} "
            f"age={row.get('age')} score={row.get('score')}"
        )

    def on_edit(ev: CellEditEvent) -> None:
        edited_label.text = f"Last edit: row_id={ev.row_id} field={ev.field} {ev.old_value!r} -> {ev.new_value!r}"
        logger.debug("[demo] cell edit: %s", ev)

    grid.on_selection_changed(on_sel)
    grid.on_cell_value_changed(on_edit)

    ui.separator()

    with ui.row().classes("items-center gap-2"):
        ui.button("Show data A", on_click=lambda: grid.set_data(_make_data_a(), reason="show_a"))
        ui.button("Show data B", on_click=lambda: grid.set_data(_make_data_b(), reason="show_b"))
        ui.button("Reset saved column state", on_click=grid.reset_column_state)

    with ui.row().classes("items-center gap-2"):
        def shuffle_rows() -> None:
            rows = list(getattr(grid, "_rows", []))  # demo-only
            random.shuffle(rows)
            grid.set_data(rows, reason="shuffle_rows")

        ui.button("Shuffle row order", on_click=shuffle_rows)

    with ui.row().classes("items-center gap-2"):
        ui.button("Select id=10", on_click=lambda: grid.select_row(10, ensure_visible=True))
        ui.button("Select id=20", on_click=lambda: grid.select_row(20, ensure_visible=True))
        ui.button("Clear selection", on_click=grid.clear_selection)

    ui.label(
        "Checklist: resize+sort, then Show data A/B → widths/sort persist. "
        "Double-click Name/Status to edit → 'Last edit' updates. "
        "ArrowUp/Down should move selection; while editing Status dropdown, arrows should NOT hijack the editor."
    ).classes("text-xs opacity-80 mt-2")


if __name__ in {"__main__", "__mp_main__"}:
    build_ui()
    ui.run(native=True, reload=False, window_size=(1000, 800), port=8001)