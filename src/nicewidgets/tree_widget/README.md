# `nicewidgets.tree_widget`

Generic NiceGUI AG Grid **Enterprise** tree widget with:

- caller-defined row identity (`row_id_field`)
- caller-defined hierarchy path field (`path_field`)
- full refresh (`set_data`) and targeted patch (`update_row`)
- subtree replacement transaction (`replace_group_rows`)
- programmatic selection (`set_selected_row_ids`, `clear_selection`)
- optional keyboard row navigation (ArrowUp / ArrowDown)
- extensible right-click context menu

## Imports (no package re-exports)

```python
from nicewidgets.aggrid_common.column_def import ColumnDef
from nicewidgets.tree_widget.config import TreeWidgetConfig
from nicewidgets.tree_widget.tree_widget import TreeWidget
```

## Core constructor

```python
tree = TreeWidget(
    columns=columns,
    row_id_field='row_id',
    rows=rows,
    on_row_selected=on_row_selected,      # optional
    on_build_context_menu=build_menu,     # optional
    config=TreeWidgetConfig(...),         # optional
    grid_options={},                      # optional AG Grid escape hatch
    path_field='hierarchy_path',          # optional
    auto_group_column_def=None,           # optional
)
```

## `TreeWidgetConfig`

- `selection_mode`: `'none' | 'single' | 'multiple'`
- `clear_selection_on_set_data`: default `True`
- `enable_keyboard_row_nav`: default `True`
- `auto_size_columns`: forwarded to `ui.aggrid(...)`
- `fit_columns_on_grid_resize`: default `False`
- `cell_font_size_px`: optional font scaling
- `row_height`, `header_height`: optional fixed pixel sizes
- `extra_grid_options`: merged before `grid_options`
- `enterprise_module_url`: default AG Grid Enterprise CDN ESM URL
- `show_index_column`: when true, prepend a synthetic 1-based Index column for
  top-level tree rows only (`node.level === 0`); child rows are blank
- `index_field`: column field name for the index column (default `file_row_index`)
- `index_header`: column header label (default `''`, blank header)
- `index_menu_label`: context-menu label when header is blank (default `Index`)
- Index values are computed in AG Grid via `valueGetter` (`forEachNode` + level 0),
  not stored in row data; column is non-sortable and width uses
  `font_scaled_column_width_px(cell_font_size_px)` (default `6 * font`, min 36px)

## Row identity and path rules

- `row_id_field` must exist in each row and be a unique non-empty `str`.
- `path_field` should contain list-like hierarchy paths compatible with AG Grid
  `treeData` + `getDataPath`.

## Notes

- `TreeWidget` sets `modules='enterprise'` on AG Grid and can configure module
  source URL via `TreeWidgetConfig.enterprise_module_url`.
- When `auto_group_column_def` is `None`, the widget sets
  `groupDisplayType: 'custom'` so callers can host disclosure triangles on one
  of their own columns with `cellRenderer: 'agGroupCellRenderer'`.
  - The widget also auto-injects `showRowGroup: True` on that column. AG Grid
    requires both `cellRenderer: 'agGroupCellRenderer'` AND `showRowGroup: True`
    for chevrons to render in a caller-defined column under
    `groupDisplayType: 'custom'`.

## Caller responsibility: shaping rows from your domain

`TreeWidget` is **domain-agnostic**. It accepts plain `list[dict[str, Any]]`
rows where each row carries:

- a unique value at `row_id_field`,
- a list-like value at `path_field` describing the hierarchy
  (e.g. `["files/foo.tif"]` for a depth-1 row, `["files/foo.tif", "<analysis-id>"]`
  for a depth-2 row),
- whatever per-column field values your `ColumnDef` list expects.

The widget **does not** know about your domain types (files, analyses,
ROIs, etc.) and provides no row-builder configuration. Callers shape rows
themselves from their domain objects. The canonical adapter pattern lives
in `sandbox/demo_tree_app/demo_tree_app.py`:

```python
def _file_tree_row(file: FakeAcqImage) -> dict[str, Any]:
    return {
        'row_id': file.file_path,
        'hierarchy_path': [file.file_path],
        'row_type': 'File',
        **file.as_row_dict(),
    }


def _analysis_tree_row(file, channel, roi_id, type_value, ts) -> dict[str, Any]:
    aid = f'{file.file_path}::ch_{channel}::roi_{roi_id}::type_{type_value}'
    return {
        'row_id': aid,
        'hierarchy_path': [file.file_path, aid],
        'row_type': 'Analysis',
        ...
    }


def _all_tree_rows(domain_collection) -> list[dict[str, Any]]:
    rows = []
    for file in domain_collection:
        rows.append(_file_tree_row(file))
        for ... in file.analyses:
            rows.append(_analysis_tree_row(...))
    return rows


tree = TreeWidget(columns=cols, row_id_field='row_id', rows=_all_tree_rows(coll))
```

Such adapters belong in your application layer (or as methods on your
domain types) — not inside `nicewidgets`. They embed domain-specific
concerns (id encoding, depth-1 vs depth-2 field naming, tree iteration
order, sentinel values for "this field doesn't apply at this depth") that
must not leak into the widget.
