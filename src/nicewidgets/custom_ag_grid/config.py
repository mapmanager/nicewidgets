# src/nicewidgets/custom_ag_grid/config.py
# gpt 20260106: add GridConfig.row_id_field for stable row IDs + programmatic selection

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional


EditorType = Literal["auto", "text", "select", "checkbox"]
SelectionMode = Literal["none", "single", "multiple"]


@dataclass
class ColumnConfig:
    """Declarative configuration for a single AG Grid column.

    Attributes:
        field: Key used in the row dictionaries (e.g. 'city').
        header: Column label shown in the grid header. Defaults to ``field``.
        editable: Whether the column is editable.
        editor: Editor type. ``"auto"`` uses AG Grid's default editor,
            ``"text"`` forces a simple text editor, and ``"select"`` uses
            the AG Grid select editor with the given ``choices``.
        choices: For ``editor="select"`` columns:
            - If an iterable, it is treated as an explicit list of allowed
              values (converted to strings).
            - If the string ``"unique"``, unique values are inferred from
              the data for this column.
            - If ``None``, the column is still editable but has no preset
              choices.
        sortable: Whether the column is sortable.
        filterable: Whether the column is filterable.
        resizable: Whether the column is resizable.
        extra_grid_options: Arbitrary additional options passed directly
            into the AG Grid column definition dictionary.
    """

    field: str
    header: Optional[str] = None
    editable: bool = False
    editor: EditorType = "auto"
    choices: Optional[Iterable[Any] | Literal["unique"]] = None

    sortable: bool = True
    filterable: bool = False
    resizable: bool = True

    extra_grid_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GridConfig:
    """Declarative configuration for grid-level AG Grid options.

    Attributes:
        selection_mode: Row selection behavior: ``"none"``, ``"single"``,
            or ``"multiple"``.
        height: CSS height for the grid container (e.g. '20rem', '400px').
        theme_class: AG Grid theme CSS class (e.g. 'ag-theme-alpine').
        zebra_rows: Whether to use alternating row background colors.
        hover_highlight: Whether to highlight rows on mouse hover.
        tight_layout: Whether to reduce padding and font size slightly.
        row_height: Pixel height of each data row.
        header_height: Pixel height of the header row.
        stop_editing_on_focus_loss: Whether editing should stop when the
            grid loses focus.
        row_id_field: (gpt 20260106) Optional field name used as a stable row id.
            If set, AG Grid uses it for row identity (enables programmatic selection
            via run_row_method and makes selection robust across data refreshes).
        show_row_index: If True, prepend a non-editable index column that tracks
            the current displayed row order (after sort/filter).
        row_index_header: Column header label for the index column.
        row_index_width: Pixel width for the index column.
        row_index_resizable: Whether the row index column can be resized by the user.
        extra_grid_options: Arbitrary additional options merged into the
            AG Grid options dictionary.
    """

    selection_mode: SelectionMode = "none"
    # height: str = "20rem"
    # height: str = "h-full"
    # height: str = '100%'
    # height: str = "h-96"

    theme_class: str = "ag-theme-alpine"

    zebra_rows: bool = True
    hover_highlight: bool = False
    tight_layout: bool = True

    row_height: int = 28
    header_height: int = 30

    stop_editing_on_focus_loss: bool = True

    # gpt 20260106: stable row id support (e.g. use "path" for file tables)
    row_id_field: Optional[str] = None

    show_row_index: bool = False
    row_index_header: str = "Idx"
    row_index_width: int = 50  # need 60 to fit index >= 100
    row_index_resizable: bool = True

    extra_grid_options: dict[str, Any] = field(default_factory=dict)