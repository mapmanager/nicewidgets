# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/demo_custom_ag_grid_v3_app.py
from __future__ import annotations

import random
from typing import Any, Dict, List

from nicegui import ui

from nicewidgets.custom_ag_grid_v3.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid_v3.custom_ag_grid_v3 import CustomAgGrid_v3
from nicewidgets.utils.logging import get_logger, setup_logging


def _make_data_a(n: int = 25) -> List[Dict[str, Any]]:
    return [{"id": i, "name": f"Row {i}", "value": i * 10, "group": "A"} for i in range(1, n + 1)]


def _make_data_b(n: int = 25) -> List[Dict[str, Any]]:
    rng = random.Random(123)
    return [{"id": i, "name": f"Item {i}", "value": rng.randint(0, 999), "group": "B"} for i in range(1, n + 1)]


def build_ui() -> None:
    setup_logging(level="DEBUG")
    logger = get_logger(__name__)

    ui.label("CustomAgGrid_v3 demo (A/B swap + preserve column state)").classes("text-lg font-semibold")

    data_a = _make_data_a(25)
    data_b = _make_data_b(25)

    columns = [
        ColumnConfig(field="id", header="ID"),
        ColumnConfig(field="name", header="Name"),
        ColumnConfig(field="value", header="Value"),
        ColumnConfig(field="group", header="Group"),
    ]

    cfg = GridConfig(selection_mode="single")

    grid = CustomAgGrid_v3(
        data_a,
        columns,
        cfg,
        allow_column_reorder=False,
        height_px=300,
        enable_keyboard_row_nav=True,
        suppress_row_hover_highlight=True,
        suppress_cell_focus=False,  # keep focus enabled for keyboard nav
    )

    def on_sel(row: Dict[str, Any] | None) -> None:
        logger.debug("[demo] selection row=%r", row)

    grid.on_selection_changed(on_sel)

    with ui.row().classes("gap-2 items-center"):
        ui.button("Show data A", on_click=lambda: grid.set_data(data_a, reason="show_a"))
        ui.button("Show data B", on_click=lambda: grid.set_data(data_b, reason="show_b"))
        ui.button("Reset saved column state", on_click=grid.reset_column_state)

    ui.label(
        "Instructions: click a row, then ArrowUp/ArrowDown should move selection (not scroll). "
        "Resize/sort columns then click Show data A/B to confirm column state persists."
    ).classes("text-sm opacity-80 mt-2")


if __name__ in {"__main__", "__mp_main__"}:
    build_ui()
    ui.run(native=True, reload=False, window_size=(2000, 520))