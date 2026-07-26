"""Composable NiceGUI AG Grid table widget.

Import from concrete submodules (this package does not re-export symbols):

```python
from nicewidgets.table_widget.column_def import ColumnDef
from nicewidgets.table_widget.config import TableWidgetConfig
from nicewidgets.table_widget.table_widget import TableWidget
```

Core features
-------------

- caller-defined, unique string row identity (``row_id_field``)
- id-based row mutation (``upsert_row``, ``update_row``, ``remove_row``)
- programmatic selection and user selection callbacks
- double-click cell editing and changed-value callbacks
- optional synthetic 1-based index column
- extensible right-click context menu and column visibility controls
- optional AG Grid Enterprise row grouping via
  :attr:`TableWidgetConfig.row_group_fields`

Community and Enterprise modes
------------------------------

With empty ``row_group_fields`` (the default), ``TableWidget`` uses AG Grid
Community. Supplying one or more fields loads the configured Enterprise ESM
module and groups by those category columns in order. NiceWidgets does not
provide an AG Grid Enterprise production license; host applications are
responsible for licensing.

The ``grid_options`` constructor argument remains the power-user escape hatch
for valid AG Grid options not represented by :class:`TableWidgetConfig`.

See ``examples/table_widget/demo_app.py`` for a runnable grouped table with an
index column, selection, editing, and a custom context menu.
"""
