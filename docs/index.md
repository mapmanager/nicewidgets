---
hide:
  - toc
---

# NiceWidgets

NiceWidgets is a Python package of reusable **NiceGUI** widgets for scientific
and desktop applications. It provides Plotly-based raster viewers and plot
widgets, AG Grid tables and trees, image toolbars, uploads, and related UI
building blocks.

NiceWidgets is **not** the CloudScope application. CloudScope is one consumer of
these widgets. Runtime code in this package does not import CloudScope or
AcqStore.

## Install

```bash
git clone https://github.com/mapmanager/nicewidgets.git
cd nicewidgets
uv sync
```

## Quick start

```python
from nicewidgets.gui_defaults import setUpGuiDefaults
from nicewidgets.aggrid_common.column_def import ColumnDef
from nicewidgets.table_widget.config import TableWidgetConfig
from nicewidgets.table_widget.table_widget import TableWidget

setUpGuiDefaults()
table = TableWidget(
    rows=[{'id': 'a', 'label': 'Sample'}],
    column_defs=[
        ColumnDef(field='id', headerName='ID'),
        ColumnDef(field='label', headerName='Label'),
    ],
    row_id_field='id',
    config=TableWidgetConfig(selection_mode='single'),
)
```

Import from concrete submodules. The package root does not re-export symbols.

## Next steps

- [Package overview](guide/overview.md)
- [Examples](guide/examples.md)
- [Widget API notes](api/widget-api.md)
