"""Runnable NiceGUI demo for ``TableWidget`` with selection/editing examples."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from nicewidgets.table_widget.column_def import ColumnDef
from nicewidgets.table_widget.config import TableWidgetConfig
from nicewidgets.table_widget.table_widget import TableWidget

ROW_ID_FIELD = 'path'


def make_demo_column_defs() -> list[ColumnDef]:
    """Build demo columns with one editable field."""
    return [
        ColumnDef(field='path', headerName='Path'),
        ColumnDef(field='kind', headerName='Kind', extra={'editable': True}),
        ColumnDef(field='size_bytes', headerName='Size (bytes)', hide=True),
    ]


def make_demo_rows() -> list[dict[str, Any]]:
    """Build synthetic rows."""
    return [
        {'path': '/lab/slide_a.tif', 'kind': 'tiff', 'size_bytes': 1_024_000},
        {'path': '/lab/slide_b.czi', 'kind': 'czi', 'size_bytes': 8_000_000},
        {'path': '/lab/kymograph.oir', 'kind': 'oir', 'size_bytes': 512_000},
    ]


@ui.page('/')
def home_page() -> None:
    """NiceGUI home page: single demo table."""
    with ui.column().classes('w-full max-w-5xl mx-auto p-4 gap-4'):
        ui.label('TableWidget demo').classes('text-h5')
        ui.label('Right-click for columns/menu. ArrowUp/ArrowDown changes selection. Double-click editable cell to edit.')
        selected = ui.label('Selected: (none)')
        edited = ui.label('Edited: (none)')

        def on_row_selected(row: dict[str, Any]) -> None:
            selected.text = f"Selected: {row.get(ROW_ID_FIELD, row)!r}"

        def on_cell_edited(row_id: str, field: str, old_value: Any, new_value: Any, _row: dict[str, Any]) -> None:
            edited.text = f'Edited: row_id={row_id!r} field={field!r} {old_value!r} -> {new_value!r}'

        def on_build_context_menu(table: TableWidget) -> None:
            ui.menu_item('Select first row', on_click=lambda: table.set_selected_row_ids(['/lab/slide_a.tif']))
            ui.menu_item('Clear selection', on_click=table.clear_selection)

        table = TableWidget(
            make_demo_column_defs(),
            ROW_ID_FIELD,
            make_demo_rows(),
            on_row_selected=on_row_selected,
            on_cell_edited=on_cell_edited,
            on_build_context_menu=on_build_context_menu,
            config=TableWidgetConfig(selection_mode='single', enable_keyboard_row_nav=True),
        )
        with ui.column().classes('w-full').style('height: 420px;'):
            table.build()

        with ui.row().classes('gap-2'):
            ui.button('Select first row', on_click=lambda: table.set_selected_row_ids(['/lab/slide_a.tif']))
            ui.button('Clear selection', on_click=table.clear_selection)


def main() -> None:
    """Entry point for ``uv run python examples/table_widget/demo_app.py``."""
    ui.run(title='nicewidgets TableWidget demo', port=8080, reload=False)


if __name__ == '__main__':
    main()
