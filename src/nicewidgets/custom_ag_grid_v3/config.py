# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/config.py
#
# Declarative config objects for CustomAgGrid_v3.
# Keep this file v3-specific so the widget can evolve without breaking older grids.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional


EditorType = Literal["auto", "text", "select"]
SelectionMode = Literal["none", "single", "multiple"]


@dataclass
class ColumnConfig:
    """Declarative configuration for a single AG Grid column.

    Attributes:
        field: Key used in the row dictionaries (e.g. 'name').
        header: Column label shown in the grid header. Defaults to ``field``.
        editable: Whether the column is editable (double-click by default).
        editor: Editor type:
            - "auto": AG Grid default editor
            - "text": force AG Grid text editor
            - "select": AG Grid select editor using ``choices``
        choices: For ``editor="select"`` columns: an iterable of allowed values (strings recommended).
        sortable: Whether the column is sortable.
        filterable: Whether the column is filterable.
        resizable: Whether the column is resizable.
        extra_grid_options: Arbitrary additional options merged into the AG Grid columnDef.
    """

    field: str
    header: Optional[str] = None

    editable: bool = False
    editor: EditorType = "auto"
    choices: Optional[Iterable[Any]] = None

    sortable: bool = True
    filterable: bool = False
    resizable: bool = True

    extra_grid_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GridConfig:
    """Declarative configuration for grid-level behavior."""

    selection_mode: SelectionMode = "none"

    # NiceGUI exposes the theme as `grid.theme` (native-safe). We don't set it here.
    # (You observed: grid.grid.theme == 'quartz' on macOS.)
    stop_editing_on_focus_loss: bool = True