---
hide:
  - toc
---

# NiceWidgets

NiceWidgets is a Python package of reusable [NiceGUI](https://nicegui.io/)
widgets for scientific and desktop applications. It provides JavaScript image
viewers and plot widgets, AG Grid tables and trees, image toolbars,
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
| [`raster_viewer_widget`](widgets/raster_viewer_widget.md) | Canvas raster viewer for multidimensional, multichannel arrays |
| [`raster_viewer`](widgets/raster_viewer.md) | Older standalone Plotly raster viewer, ROI and trace overlays |
| [`plotly_plot`](widgets/plotly_plot.md) | General Plotly line/measurement plot widget |
| [`table_widget`](widgets/table_widget.md) / [`tree_widget`](widgets/tree_widget.md) | AG Grid table and tree wrappers |
| [`nicepool`](widgets/nicepool.md) | DataFrame-driven plot pool UI |
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

- [Widgets](widgets/index.md)
- [Examples](guide/examples.md)
- [Layout and sizing](guide/layout-and-sizing.md)
- [Widget API notes](api/widget-api.md)
