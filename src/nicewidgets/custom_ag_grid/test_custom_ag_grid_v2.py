from nicegui import ui
from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid.custom_ag_grid_v2 import CustomAgGrid_v2

@ui.page("/")
def index():
    rows_a = [{"id": f"row_{i:03d}", "name": f"Item {i}", "value": i * 10} for i in range(1, 13)]
    rows_b = [{"id": f"alt_{i:03d}", "name": f"Alt {i}", "value": i * 7} for i in range(1, 4)]
    cols = [
        ColumnConfig("id", editable=False),
        ColumnConfig("name", editable=True),
        ColumnConfig("value", editable=True),
    ]
    cfg = GridConfig(selection_mode="single", row_id_field="id")

    grid = CustomAgGrid_v2(rows_a, cols, cfg)

    def log_selected(prefix: str) -> None:
        selected = grid.get_selected_rows()
        selected_id = selected[0]["id"] if selected else None
        print(f"[app] {prefix} selected={selected_id} count={len(selected)}")

    def set_rows(rows, label: str) -> None:
        log_selected(f"before set_data({label})")
        grid.set_data(rows)
        log_selected(f"after set_data({label})")

    with ui.row():
        ui.button("Load A (12 rows)", on_click=lambda: set_rows(rows_a, "A"))
        ui.button("Load B (3 rows)", on_click=lambda: set_rows(rows_b, "B"))
        ui.button("Clear", on_click=lambda: set_rows([], "empty"))

    grid.on_row_selected(lambda i, row: print("[app] selected", i, row["id"]))
    grid.on_cell_edited(lambda i, f, o, n, row: print("[app] edited", row["id"], f, o, "->", n))

ui.run()