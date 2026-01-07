# nicewidgets/custom_ag_grid/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

EditorType = Literal["auto", "text", "select"]
SelectionMode = Literal["none", "single", "multiple"]


@dataclass
class ColumnConfig:
    """Column definition wrapper for AG Grid.

    Attributes:
        field: Key in the row dict.
        header: Column header label.
        width: Optional pixel width.
        hide: If True, column is hidden.
        align: Optional alignment hint used to set cellClass.
        editable: If True, allow cell editing.
        editor: Optional editor hint.
        options: Optional options for select editor.
    """

    field: str
    header: str
    width: int | None = None
    hide: bool = False
    align: Literal["left", "center", "right"] | None = None
    editable: bool = False
    editor: EditorType = "auto"
    options: list[str] | None = None


@dataclass
class GridConfig:
    """Declarative configuration for grid-level AG Grid options."""

    selection_mode: SelectionMode = "single"
    height: str = "24rem"
    theme_class: str = "ag-theme-alpine"
    zebra_rows: bool = True
    hover_highlight: bool = True
    tight_layout: bool = True
    row_height: int = 28
    header_height: int = 32
    stop_editing_on_focus_loss: bool = True
    extra_grid_options: dict[str, Any] = field(default_factory=dict)

    # NEW: stable row identity (enables run_row_method and selection persistence)
    row_id_field: str | None = None
