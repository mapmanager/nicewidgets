# `nicewidgets.table_widget`

Generic NiceGUI AG Grid widget with:

- caller-defined row identity (`row_id_field`)
- id-based row mutation (`upsert_row`, `update_row`, `remove_row`)
- programmatic selection (`set_selected_row_ids`, `clear_selection`)
- edit callbacks (double-click edit + edit-stopped changed-value event)
- extensible right-click context menu

## Imports (no package re-exports)

```python
from nicewidgets.table_widget.column_def import ColumnDef
from nicewidgets.table_widget.config import TableWidgetConfig
from nicewidgets.table_widget.table_widget import TableWidget
```

## Core constructor

```python
table = TableWidget(
    columns=columns,
    row_id_field='path',
    rows=rows,
    on_row_selected=on_row_selected,     # optional
    on_cell_edited=on_cell_edited,       # optional
    on_build_context_menu=build_menu,    # optional
    config=TableWidgetConfig(...),       # optional
    grid_options={},                     # optional AG Grid escape hatch
)
```

## `TableWidgetConfig`

- `selection_mode`: `'none' | 'single' | 'multiple'`
- `clear_selection_on_set_data`: default `True` (full refresh clears selection)
- `enable_edit_on_double_click`: default `True`
- `enable_keyboard_row_nav`: default `True` (ArrowUp/ArrowDown)
- `stop_editing_when_cells_lose_focus`: default `True`
- `auto_size_columns`: forwarded to `ui.aggrid(...)`
- `extra_grid_options`: merged before `grid_options`

## Row identity rules

- `row_id_field` must be a non-empty string key present in every row.
- id value must be a unique non-empty `str`.
- client-side AG Grid identity is wired via dynamic `':getRowId'`.

## Context menu extensibility

Built-in menu always includes column visibility checkboxes.
If `on_build_context_menu(table_widget)` is provided, a separator is added and callers can append custom menu items.

## Demo app

```bash
uv run python examples/table_widget/demo_app.py
```

## Notes

- `grid_options` is a power-user override for AG Grid-specific features not exposed as first-class widget args.
- For strict package boundaries, keep `nicewidgets` free of `acqstore` imports in committed code.
