"""Column definition shared by AG Grid-backed widgets (table, tree, ...)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ColumnDef:
    """Minimal column metadata mapped to AG Grid ``columnDefs`` entries.

    Attributes:
        field: Row dict key and AG Grid ``field``.
        headerName: Column header label.
        hide: When ``True``, the column starts hidden (AG Grid ``hide``).
        extra: Additional AG Grid colDef keys merged after the built-in keys
            (caller must supply valid AG Grid option names and JSON-serializable
            values unless using NiceGUI dynamic ``:`` keys).
    """

    field: str
    headerName: str
    hide: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def as_aggrid_column_def(self) -> dict[str, Any]:
        """Return one AG Grid column definition dict.

        Returns:
            Column definition suitable for ``ui.aggrid`` ``columnDefs``.
        """
        col: dict[str, Any] = {
            'field': self.field,
            'headerName': self.headerName,
            'hide': self.hide,
        }
        col.update(dict(self.extra))
        return col
