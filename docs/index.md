---
hide:
  - toc
---

# NiceWidgets

NiceWidgets is a Python package of reusable [NiceGUI](https://nicegui.io/)
widgets for scientific and desktop applications. It provides Plotly-based
raster viewers and plot widgets, AG Grid tables and trees, image toolbars,
uploads, and related UI building blocks.

[CloudScope](https://mapmanager.github.io/cloudscope/) is an example GUI
application that uses NiceWidgets.

## Install

```bash
git clone https://github.com/mapmanager/nicewidgets.git
cd nicewidgets
uv sync
```

Optional desktop extras (`pyperclip`, `pyperclipimg`, `pywebview`):

```bash
uv sync --extra desktop
```

## Available widgets

| Area | Role |
|------|------|
| `raster_viewer` | Plotly multiresolution raster viewer, ROI and trace overlays |
| `plotly_plot` | General Plotly line/measurement plot widget |
| `echart_widget` | ECharts line/plot widget |
| `table_widget` / `tree_widget` | AG Grid table and tree wrappers |
| `nicepool` | DataFrame-driven plot pool UI |
| `image_toolbar_widget` / `contrast_widget` | Image toolbar and contrast controls |
| `upload_widget` | File upload normalization |
| `utils` | Logging, clipboard, desktop detection helpers |

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

## Next steps

- [Examples](guide/examples.md)
- [Widget API notes](api/widget-api.md)
