# run_grid_gpt.py

from nicegui import ui

from nicewidgets.custom_ag_grid import CustomAgGrid
from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig, SelectionMode  # adjust import to your package layout


def main() -> None:
    # Sample data
    rows = [
        {"id": 1, "name": "Alice", "age": 30, "status": "active"},
        {"id": 2, "name": "Bob", "age": 42, "status": "inactive"},
        {"id": 3, "name": "Charlie", "age": 25, "status": "active"},
    ]

    # Column configuration
    columns = [
        ColumnConfig(field="id", header="ID", editable=False),
        ColumnConfig(field="name", header="Name", editable=True),
        ColumnConfig(field="age", header="Age", editable=True),
        ColumnConfig(
            field="status",
            header="Status",
            editable=True,
            editor="select",
            choices=["active", "inactive", "pending"],
        ),
    ]

    grid_config = GridConfig(
        selection_mode="single",  # SelectionMode type alias / Literal["none","single","multiple"]
        height="h-96",
    )

    @ui.page("/")
    def index() -> None:
        ui.label("CustomAgGrid demo (grid_gpt.py)").classes("text-2xl font-bold mb-4")

        custom_grid = CustomAgGrid(
            data=rows,
            columns=columns,
            grid_config=grid_config,
        )

        # Register event handlers
        def handle_cell_edited(
            row_index: int,
            field: str,
            old_value,
            new_value,
            row_data: dict,
        ) -> None:
            print(
                f"[CELL EDITED] row={row_index}, field={field}, "
                f"old={old_value!r}, new={new_value!r}, row={row_data}"
            )

        def handle_row_selected(row_index: int, row_data: dict) -> None:
            print(f"[ROW SELECTED] row={row_index}, row={row_data}")

        custom_grid.on_cell_edited(handle_cell_edited)
        custom_grid.on_row_selected(handle_row_selected)

        ui.label(
            "Try editing cells or selecting rows; events will be printed in the console."
        ).classes("mt-4 text-sm text-gray-500")

    ui.run(port=8003, reload=True)


if __name__ in {"__main__", "__mp_main__"}:
    main()
