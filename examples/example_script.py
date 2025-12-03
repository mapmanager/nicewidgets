from nicegui import ui
import pandas as pd
from kymflow.v2.gui.nicegui.custom_ag_grid.config import ColumnConfig, GridConfig
from kymflow.v2.gui.nicegui.custom_ag_grid.grid import CustomAgGrid

df = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"],
        "city": ["Sacramento", "Baltimore", "Montreal"],
        "score": [88.5, 92.0, 76.5],
    }
)

columns = [
    ColumnConfig("id", header="ID"),
    ColumnConfig("name", header="Name", editable=True),
    ColumnConfig("city", header="City", editable=True, editor="select", choices="unique"),
    ColumnConfig("score", header="Score"),
]

grid_cfg = GridConfig(selection_mode="single", height="18rem")

with ui.header().classes("py-2 px-4"):
    ui.label("CustomAgGrid demo")

grid = CustomAgGrid(data=df, columns=columns, grid_config=grid_cfg)

def on_edit(row: int, field: str, old: object, new: object, row_data: dict) -> None:
    print("EDIT:", row, field, old, "->", new, row_data)

def on_select(row: int, row_data: dict) -> None:
    print("SELECT:", row, row_data)

grid.cell_edited.connect(on_edit)
grid.row_selected.connect(on_select)

ui.run()
