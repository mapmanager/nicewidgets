# src/nicewidgets/custom_ag_grid/custom_ag_grid_v2.py
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Optional, TYPE_CHECKING, Union

from nicegui import events, ui

from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig, SelectionMode
from nicewidgets.custom_ag_grid.js_hooks import (
    js_on_cell_double_clicked_start_editing,
    js_on_cell_editing_stopped_emit_change,
    js_on_cell_key_down_select_prev_next,
    js_on_row_clicked,
)
from nicewidgets.custom_ag_grid.theme import ensure_aggrid_theme
from nicewidgets.custom_ag_grid.styles import AGGRID_HIDE_CELL_FOCUS_CSS
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

# Optional pandas
try:  # pragma: no cover
    import pandas as _pd  # type: ignore[import]
    HAS_PANDAS = True
except Exception:  # pragma: no cover
    _pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

# Optional polars
try:  # pragma: no cover
    import polars as _pl  # type: ignore[import]
    HAS_POLARS = True
except Exception:  # pragma: no cover
    _pl = None  # type: ignore[assignment]
    HAS_POLARS = False

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
else:
    pd = _pd  # type: ignore[assignment]
    pl = _pl  # type: ignore[assignment]

RowDict = dict[str, Any]
RowsLike = list[RowDict]
DataLike = Union[RowsLike, "pd.DataFrame", "pl.DataFrame"]  # type: ignore[name-defined]

CellEditedHandler = Callable[[int, str, Any, Any, RowDict], None]
RowSelectedHandler = Callable[[int, RowDict], None]


class CustomAgGrid_v2:
    """NiceGUI AG Grid wrapper with robust selection, keyboard navigation, and cell editing.

    Key behavior:
    - Row selection:
        - mouse click selects row
        - ArrowUp/ArrowDown selects previous/next displayed row
        - selection callback is deduped (fires once per selection change)
    - Cell editing:
        - double-click starts editing cell
        - edit-finished event fires once when editing stops AND value changed

    Why v2 uses JS hooks:
    NiceGUI's `.on('cellKeyDown')` / `.on('rowSelected')` payloads are often
    sanitized (DOM KeyboardEvent stripped). AG Grid native callbacks in
    gridOptions provide full params (including params.event.key). We emit
    small JSON payloads back to Python via `emitEvent(...)`.

    Public API:
        on_row_selected(handler): handler(row_index, row_dict)
        on_cell_edited(handler): handler(row_index, field_name, old, new, row_dict)
        set_data(data): replace grid rows
        get_selected_rows(): last known selected rows (at most one for single select)
        set_selected_row_ids(row_ids): programmatic selection (requires row_id_field)
    """

    def __init__(
        self,
        data: DataLike,
        columns: list[ColumnConfig],
        grid_config: GridConfig | None = None,
        parent: ui.element | None = None,
        *,
        enable_keyboard_row_nav: bool = True,
        enable_edit_on_double_click: bool = True,
        inject_hide_cell_focus_css: bool = True,
    ) -> None:
        ensure_aggrid_theme()

        self._rows: RowsLike = self._convert_input_to_rows(data)
        self._columns: list[ColumnConfig] = columns
        self._grid_config: GridConfig = grid_config or GridConfig()

        if not self._grid_config.row_id_field:
            raise ValueError("CustomAgGrid_v2 requires GridConfig.row_id_field for robust selection/editing.")

        self._row_id_field: str = self._grid_config.row_id_field

        # Instance-unique emitted event names (avoid collisions across multiple grids)
        self._evt_select: str = f"custom_aggrid_v2_select_{id(self)}"
        self._evt_edit: str = f"custom_aggrid_v2_edit_{id(self)}"

        # Dedup selection changes by rowId
        self._last_selected_row_id: Optional[str] = None

        # last known selected rows
        self._last_selected_rows: RowsLike = []

        # feedback loop guard for programmatic selection
        self._selection_origin: str = "internal"

        # callback registries
        self._cell_edited_handlers: list[CellEditedHandler] = []
        self._row_selected_handlers: list[RowSelectedHandler] = []

        # container classes
        container_classes = f"w-full {self._grid_config.height} {self._grid_config.theme_class}"
        if self._grid_config.zebra_rows:
            container_classes += " aggrid-zebra"
        if self._grid_config.hover_highlight:
            container_classes += " aggrid-hover"
        if self._grid_config.tight_layout:
            container_classes += " aggrid-tight"

        self._container: ui.element = parent or ui.column()
        self._container.classes(container_classes)

        if inject_hide_cell_focus_css:
            # Hide focused-cell visuals without breaking keyboard nav
            ui.add_head_html(f"<style>{AGGRID_HIDE_CELL_FOCUS_CSS}</style>")

        # Build options + create grid
        column_defs = self._build_column_defs()
        grid_options = self._build_grid_options(
            column_defs=column_defs,
            row_data=self._rows,
            enable_keyboard_row_nav=enable_keyboard_row_nav,
            enable_edit_on_double_click=enable_edit_on_double_click,
        )

        self._grid: ui.aggrid = ui.aggrid(grid_options)

        # Register Python listeners for emitted events
        ui.on(self._evt_select, self._on_select_emitted)
        ui.on(self._evt_edit, self._on_edit_emitted)

        # Optional: keep a simple debug click hook (safe payload, helpful during dev)
        # You can remove later.
        self._grid.on("cellClicked", lambda e: logger.debug(f"cellClicked args={e.args}"))

        logger.info(
            "CustomAgGrid_v2 initialized: rows=%s cols=%s selection=%s row_id_field=%s",
            len(self._rows),
            len(self._columns),
            self._grid_config.selection_mode,
            self._row_id_field,
        )

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def grid(self) -> ui.aggrid:
        """Underlying NiceGUI `ui.aggrid` element (escape hatch)."""
        return self._grid

    @property
    def rows(self) -> RowsLike:
        """Shallow copy of current rows."""
        return [r.copy() for r in self._rows]

    def get_selected_rows(self) -> RowsLike:
        """Return last known selected rows (at most 1 for single selection)."""
        return [r.copy() for r in self._last_selected_rows]

    # ------------------------------------------------------------------
    # Public event registration
    # ------------------------------------------------------------------

    def on_cell_edited(self, handler: CellEditedHandler) -> None:
        """Register handler(row_index, field_name, old_value, new_value, full_row_dict)."""
        self._cell_edited_handlers.append(handler)

    def on_row_selected(self, handler: RowSelectedHandler) -> None:
        """Register handler(row_index, full_row_dict)."""
        self._row_selected_handlers.append(handler)

    # ------------------------------------------------------------------
    # Data replacement
    # ------------------------------------------------------------------

    def set_data(self, data: DataLike) -> None:
        """Replace grid data and refresh."""
        self._rows = self._convert_input_to_rows(data)
        self._grid.options["rowData"] = self._rows
        self._grid.update()

        # Reset selection tracking (caller can re-select programmatically if desired)
        self._last_selected_rows = []
        self._last_selected_row_id = None

    # ------------------------------------------------------------------
    # Programmatic selection
    # ------------------------------------------------------------------

    def set_selected_row_ids(self, row_ids: list[str], *, origin: str = "external") -> None:
        """Programmatically select rows by rowId.

        Notes:
        - requires GridConfig.row_id_field
        - for single-select, only the first rowId is used
        """
        self._selection_origin = origin

        if self._grid_config.selection_mode in {"single", "none"}:
            self._grid.run_grid_method("deselectAll")

        if not row_ids:
            self._selection_origin = "internal"
            return

        if self._grid_config.selection_mode == "single":
            row_ids = row_ids[:1]

        for i, rid in enumerate(row_ids):
            clear = True if (i == 0 and self._grid_config.selection_mode == "single") else False
            self._grid.run_row_method(str(rid), "setSelected", True, clear)

        self._selection_origin = "internal"

    # ------------------------------------------------------------------
    # Internal: conversion
    # ------------------------------------------------------------------

    def _convert_input_to_rows(self, data: DataLike) -> RowsLike:
        if isinstance(data, list):
            if all(isinstance(row, Mapping) for row in data):
                return [dict(row) for row in data]
            raise TypeError("List input must contain mapping/dict-like rows.")

        if HAS_PANDAS and pd is not None and isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")

        if HAS_POLARS and pl is not None and isinstance(data, pl.DataFrame):
            return data.to_dicts()

        raise TypeError("Unsupported data type: expected list[dict], pandas.DataFrame, or polars.DataFrame.")

    # ------------------------------------------------------------------
    # Internal: options building
    # ------------------------------------------------------------------

    def _build_column_defs(self) -> list[dict[str, Any]]:
        defs: list[dict[str, Any]] = []
        for col in self._columns:
            col_def: dict[str, Any] = {
                "headerName": col.header or col.field,
                "field": col.field,
                "editable": col.editable,
                "sortable": col.sortable,
                "filter": col.filterable,
                "resizable": col.resizable,
            }

            if col.editor == "select" and col.editable:
                if col.choices == "unique":
                    seen: set[str] = set()
                    for row in self._rows:
                        v = row.get(col.field)
                        if v is not None:
                            seen.add(str(v))
                    values = sorted(seen)
                elif col.choices is not None:
                    values = [str(v) for v in col.choices]
                else:
                    values = []

                col_def[":cellEditor"] = "'agSelectCellEditor'"
                col_def["cellEditorParams"] = {"values": values}

            col_def.update(col.extra_grid_options)
            defs.append(col_def)

        return defs

    def _row_selection_object(self, selection: SelectionMode) -> Optional[dict[str, Any]]:
        """AG Grid v32+ rowSelection object (avoids deprecation warning)."""
        if selection == "none":
            return None
        if selection == "single":
            return {"mode": "singleRow", "enableClickSelection": True, "checkboxes": False}
        return {"mode": "multiRow", "enableClickSelection": True, "checkboxes": False}

    def _build_grid_options(
        self,
        *,
        column_defs: list[dict[str, Any]],
        row_data: RowsLike,
        enable_keyboard_row_nav: bool,
        enable_edit_on_double_click: bool,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {
            "columnDefs": column_defs,
            "rowData": row_data,
            "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
            "rowHeight": self._grid_config.row_height,
            "headerHeight": self._grid_config.header_height,
        }

        # stop editing on focus loss (nice UX)
        if self._grid_config.stop_editing_on_focus_loss:
            opts["stopEditingWhenCellsLoseFocus"] = True

        # stable row id
        field = self._row_id_field
        opts[":getRowId"] = f"(params) => params.data && String(params.data['{field}'])"

        # selection (AG Grid v32+ object form)
        row_sel = self._row_selection_object(self._grid_config.selection_mode)
        if row_sel is not None:
            opts["rowSelection"] = row_sel

            # Emit selection changes consistently (mouse click)
            opts[":onRowClicked"] = js_on_row_clicked(
                emit_event=self._evt_select,
                row_id_field=self._row_id_field,
            )

            # ArrowUp/Down selects prev/next displayed row
            if enable_keyboard_row_nav:
                opts[":onCellKeyDown"] = js_on_cell_key_down_select_prev_next(
                    emit_event=self._evt_select,
                    row_id_field=self._row_id_field,
                )

        # editing hooks
        if enable_edit_on_double_click:
            opts[":onCellDoubleClicked"] = js_on_cell_double_clicked_start_editing()
            opts[":onCellEditingStopped"] = js_on_cell_editing_stopped_emit_change(
                emit_event=self._evt_edit,
                row_id_field=self._row_id_field,
                include_row_data=True,
            )

        # user extra options last (allows overriding if they really want)
        opts.update(self._grid_config.extra_grid_options)
        return opts

    # ------------------------------------------------------------------
    # Internal: emitted event handlers
    # ------------------------------------------------------------------

    def _on_select_emitted(self, e: events.GenericEventArguments) -> None:
        """Handle unified selection change events emitted from AG Grid JS hooks."""
        args: dict[str, Any] = e.args or {}

        if getattr(self, "_selection_origin", "internal") != "internal":
            # Ignore if selection triggered by our own programmatic selection with origin != internal
            return

        row_id = args.get("rowId")
        row_index = args.get("rowIndex")
        data = args.get("data") or {}

        if row_id is None or row_index is None:
            return

        row_id_str = str(row_id)

        # Dedup: only fire when the selected row actually changes
        if row_id_str == self._last_selected_row_id:
            return
        self._last_selected_row_id = row_id_str

        try:
            i = int(row_index)
        except Exception:
            return

        self._last_selected_rows = [dict(data)]

        logger.debug("selection_change rowIndex=%s rowId=%s source=%s key=%s", i, row_id_str, args.get("source"), args.get("key"))

        for handler in list(self._row_selected_handlers):
            try:
                handler(i, dict(data))
            except Exception:
                logger.exception("Error in row_selected handler")

    def _on_edit_emitted(self, e: events.GenericEventArguments) -> None:
        """Handle edit-finished events emitted from AG Grid JS hooks."""
        args: dict[str, Any] = e.args or {}

        row_index = args.get("rowIndex")
        col_id = args.get("colId")
        old_value = args.get("oldValue")
        new_value = args.get("newValue")
        data = args.get("data") or {}

        if row_index is None or col_id is None:
            return

        try:
            i = int(row_index)
        except Exception:
            return
        if not (0 <= i < len(self._rows)):
            return

        field = str(col_id)

        # Keep internal data in sync
        self._rows[i][field] = new_value

        logger.debug("cell_edit_finished row=%s field=%s %r -> %r", i, field, old_value, new_value)

        for handler in list(self._cell_edited_handlers):
            try:
                handler(i, field, old_value, new_value, dict(data))
            except Exception:
                logger.exception("Error in cell_edited handler")