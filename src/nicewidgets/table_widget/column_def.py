"""Column definition for AG Grid-backed table widgets.

Re-export of :class:`nicewidgets.aggrid_common.column_def.ColumnDef` so existing
``from nicewidgets.table_widget.column_def import ColumnDef`` imports keep
working while the dataclass itself lives under ``nicewidgets.aggrid_common``
and is shared with other AG Grid widgets (e.g. ``tree_widget``).
"""

from __future__ import annotations

from nicewidgets.aggrid_common.column_def import ColumnDef

__all__ = ['ColumnDef']
