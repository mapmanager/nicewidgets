# TableWidget

`TableWidget` wraps NiceGUI `ui.aggrid` with stable string row ids, selection,
optional editing, context menus, and optional AG Grid Enterprise row grouping.

## Embed

```python
from nicegui import ui
from nicewidgets.table_widget.column_def import ColumnDef
from nicewidgets.table_widget.config import TableWidgetConfig
from nicewidgets.table_widget.table_widget import TableWidget

table = TableWidget(
    [
        ColumnDef(field='path', headerName='Path'),
        ColumnDef(field='category', headerName='Category'),
    ],
    'path',
    [
        {'path': '/a.tif', 'category': 'Images'},
        {'path': '/b.csv', 'category': 'Tables'},
    ],
    config=TableWidgetConfig(
        selection_mode='single',
        show_index_column=True,
        row_group_fields=('category',),
    ),
    on_row_selected=lambda row: print(row),
)
with ui.column().classes('w-full').style('height: 24rem;'):
    table.build()
table.set_dark_mode(False)
```

Non-empty `row_group_fields` loads AG Grid Enterprise. NiceWidgets does not
ship a production Enterprise license.

Theme: `set_theme` / `set_dark_mode` set AG Grid `data-ag-theme-mode`.

Demo: `examples/table_widget/` (also `/table` in the combined demo).

## Configuration

`TableWidgetConfig` covers selection, editing hooks, index column, row/header
heights, grouping, and Enterprise module URL.

## API

::: nicewidgets.table_widget.table_widget.TableWidget
    options:
      show_root_heading: true
      heading_level: 3

::: nicewidgets.table_widget.config.TableWidgetConfig
    options:
      show_root_heading: true
      heading_level: 3
