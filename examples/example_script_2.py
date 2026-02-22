from nicegui import ui

from nicewidgets.utils.logging import configure_logging
from custom_ag_grid.config import ColumnConfig, GridConfig
from custom_ag_grid.grid import CustomAgGrid


def main() -> None:
    """Minimal demo of CustomAgGrid using only list-of-dicts data."""
    configure_logging(level="INFO")

    # Example rows
    rows = [
        {"id": 1, "name": "Alice", "city": "Sacramento", "score": 88.5},
        {"id": 2, "name": "Bob", "city": "Baltimore", "score": 92.0},
        {"id": 3, "name": "Carol", "city": "Montreal", "score": 76.5},
    ]

    # Column definitions
    columns = [
        ColumnConfig("id", header="ID"),
        ColumnConfig("name", header="Name", editable=True),
        ColumnConfig("city", header="City", editable=True, editor="select", choices="unique"),
        ColumnConfig("score", header="Score"),
    ]

    # Grid configuration
    grid_cfg = GridConfig(
        selection_mode="single",
        height="20rem",
        zebra_rows=True,
        hover_highlight=True,
        tight_layout=True,
    )

    # Header UI
    with ui.header().classes("py-2 px-4"):
        ui.label("CustomAgGrid – Basic Demo")

    # Create grid
    grid = CustomAgGrid(
        data=rows,
        columns=columns,
        grid_config=grid_cfg,
    )

    # Attach signals
    def on_edit(row_index: int, field: str, old_value: object, new_value: object, row: dict) -> None:
        print(f"[EDIT] row={row_index}, field={field!r}, {old_value!r} → {new_value!r}")

    def on_select(row_index: int, row: dict) -> None:
        print(f"[SELECT] row={row_index}: {row}")

    grid.cell_edited.connect(on_edit)
    grid.row_selected.connect(on_select)

    ui.run()


if __name__ in {"__main__", "__mp_main__"}:
    main()
