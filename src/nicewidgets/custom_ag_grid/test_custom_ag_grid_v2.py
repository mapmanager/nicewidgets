from nicegui import ui
from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid.custom_ag_grid_v2 import CustomAgGrid_v2

@ui.page("/")
def index():
    rows = [{"id": f"row_{i:03d}", "name": f"Item {i}", "value": i * 10} for i in range(1, 13)]
    cols = [
        ColumnConfig("id", editable=False),
        ColumnConfig("name", editable=True),
        ColumnConfig("value", editable=True),
    ]
    cfg = GridConfig(selection_mode="single", row_id_field="id")

    grid = CustomAgGrid_v2(rows, cols, cfg)

    grid.on_row_selected(lambda i, row: print("[app] selected", i, row["id"]))
    grid.on_cell_edited(lambda i, f, o, n, row: print("[app] edited", row["id"], f, o, "->", n))

ui.run()