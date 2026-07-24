# NiceWidgets

[![Tests](https://github.com/mapmanager/nicewidgets/actions/workflows/tests.yml/badge.svg)](https://github.com/mapmanager/nicewidgets/actions/workflows/tests.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

NiceWidgets is a Python package of **reusable NiceGUI widgets** for scientific and desktop applications: Plotly raster viewers, plot widgets, AG Grid tables and trees, toolbars, uploads, and related UI building blocks.

It depends on **NiceGUI**. It is **not** the CloudScope application, and it does **not** depend on CloudScope or AcqStore.

NiceWidgets was extracted from the CloudScope project as a standalone package. This repository does not preserve the original Git history.

## Install (development)

```bash
git clone https://github.com/mapmanager/nicewidgets.git
cd nicewidgets
uv sync --group dev
```

Optional desktop clipboard / native-window extras:

```bash
uv sync --group dev --extra desktop
```

## Quick start

```python
from nicewidgets.gui_defaults import setUpGuiDefaults
from nicewidgets.table_widget.config import TableWidgetConfig
from nicewidgets.table_widget.table_widget import TableWidget
from nicewidgets.aggrid_common.column_def import ColumnDef

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

## Examples

Runnable demos live under `examples/` (not installed with the wheel):

```bash
uv run python examples/table_widget/demo_app.py
uv run python examples/raster_viewer/nicegui_raster_demo.py
```

## Documentation

```bash
uv sync --group docs
uv run mkdocs serve
```

Published docs: https://mapmanager.github.io/nicewidgets/

## Tests

```bash
uv sync --group dev
uv run pytest
```

## Build the package

```bash
uv build
```

## License

GPL-3.0-only. Copyright (c) Robert Cudmore.

Repository: https://github.com/mapmanager/nicewidgets
