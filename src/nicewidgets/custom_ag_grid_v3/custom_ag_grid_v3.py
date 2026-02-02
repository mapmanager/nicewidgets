# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/custom_ag_grid_v3.py
#
# Step 2 (guardrails): add "grid ready" + queued selection if select_row() is called too early.
# No behavior changes unless select_row() happens before firstDataRendered.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Union

from nicegui import ui

from nicewidgets.custom_ag_grid_v3.config import ColumnConfig, GridConfig
from nicewidgets.custom_ag_grid_v3.js_hooks_v3 import (
    js_capture_column_state,
    js_focus_selected_row_first_col,
    js_get_row_id,
    js_on_cell_key_down_select_prev_next,
    js_on_row_clicked_focus_clicked_col,
    js_select_row_by_id,
)
from nicewidgets.custom_ag_grid_v3.theme_v3 import install_theme_v3
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

RowDict = dict[str, Any]
RowsLike = list[RowDict]
DataLike = Union[RowsLike]
RowId = Union[str, int]

SelectionCallback = Callable[[Optional[RowDict]], None]
CellEditCallback = Callable[["CellEditEvent"], None]


@dataclass(frozen=True)
class CellEditEvent:
    """Cell edit event emitted from the grid."""

    row_id: str
    field: str
    old_value: Any
    new_value: Any
    row: RowDict


class CustomAgGrid_v3:
    """
    Public API (stable):
      - set_data(rows, reason="...")               : replace rowData without recreating the grid
      - select_row(row_id, ensure_visible=True)    : select a row by id (requires row_id_field)
      - clear_selection()                          : deselect all
      - on_selection_changed(callback)             : selection callback (row dict or None)
      - on_cell_value_changed(callback)            : emits CellEditEvent on committed edit (double-click)
      - reset_column_state()                       : forget saved column widths/sorts

    Step-2 guardrail:
      - If select_row(...) is called before the grid is ready (firstDataRendered not seen yet),
        the request is queued and applied once ready.
    """

    def __init__(
        self,
        data: DataLike,
        columns: list[ColumnConfig],
        grid_config: GridConfig,
        *,
        row_id_field: str = "id",
        allow_column_reorder: bool = False,
        height_px: int = 300,
        capture_debounce_s: float = 0.05,
        restore_slow_s: float = 0.20,
        enable_keyboard_row_nav: bool = True,
        suppress_cell_focus: bool = False,
        suppress_row_hover_highlight: bool = True,
        install_css_theme: bool = True,
    ) -> None:
        if install_css_theme:
            install_theme_v3()

        self._rows: RowsLike = self._ensure_rows(data)
        self._columns = columns
        self._cfg = grid_config
        self._row_id_field = str(row_id_field)

        self._height_px = int(height_px)
        self._capture_debounce_s = float(capture_debounce_s)
        self._restore_slow_s = float(restore_slow_s)

        self._saved_column_state: list[dict[str, Any]] | None = None
        self._capture_scheduled: bool = False
        self._last_capture_reason: str = ""

        self._selection_cb: SelectionCallback | None = None
        self._cell_edit_cb: CellEditCallback | None = None

        # Step 2: ready + pending selection
        self._grid_ready: bool = False
        self._pending_select: tuple[RowId, bool] | None = None  # (row_id, ensure_visible)

        selection_mode = getattr(self._cfg, "selection_mode", "none")
        if selection_mode == "multiple":
            row_selection = {"mode": "multiRow", "checkboxes": False, "headerCheckbox": False}
        elif selection_mode == "single":
            row_selection = {"mode": "singleRow", "checkboxes": False, "headerCheckbox": False}
        else:
            row_selection = None  # selection off

        default_col_def: dict[str, Any] = {
            "sortable": True,
            "resizable": True,
            "suppressMovable": (not allow_column_reorder),
            # Explicit: keep click for selection; edit by double-click only.
            "singleClickEdit": False,
        }

        grid_options: dict[str, Any] = {
            "columnDefs": self._build_column_defs(),
            "rowData": self._rows,
            "defaultColDef": default_col_def,
            "suppressMovableColumns": (not allow_column_reorder),
            "suppressRowClickSelection": False,
            "suppressCellFocus": bool(suppress_cell_focus),
            "suppressRowHoverHighlight": bool(suppress_row_hover_highlight),
            "stopEditingWhenCellsLoseFocus": bool(getattr(self._cfg, "stop_editing_on_focus_loss", True)),
            ":getRowId": js_get_row_id(self._row_id_field),
        }
        if row_selection is not None:
            grid_options["rowSelection"] = row_selection

        if enable_keyboard_row_nav:
            grid_options[":onRowClicked"] = js_on_row_clicked_focus_clicked_col()
            grid_options[":onCellKeyDown"] = js_on_cell_key_down_select_prev_next()

        logger.debug(
            "[CustomAgGrid_v3] init rows=%d height_px=%d keynav=%s",
            len(self._rows),
            self._height_px,
            enable_keyboard_row_nav,
        )

        self.grid = ui.aggrid(grid_options).classes("w-full kym-aggrid-v3").style(
            f"height: {self._height_px}px;"
        )

        # Column state capture/restore
        self.grid.on("columnResized", lambda e: self._schedule_capture_state("columnResized"))
        self.grid.on("sortChanged", lambda e: self._schedule_capture_state("sortChanged"))
        self.grid.on("columnVisible", lambda e: self._schedule_capture_state("columnVisible"))
        self.grid.on("columnPinned", lambda e: self._schedule_capture_state("columnPinned"))

        # First data rendered: set ready + restore state + apply any queued selection
        self.grid.on("firstDataRendered", self._on_first_data_rendered)

        # Selection + editing events
        self.grid.on("selectionChanged", self._on_selection_changed_event)
        self.grid.on("cellValueChanged", self._on_cell_value_changed_event)

        self._schedule_capture_state("initial")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_data(self, data: DataLike, *, reason: str = "") -> None:
        """Replace the grid's rows without recreating the grid."""
        self._rows = self._ensure_rows(data)
        logger.debug("[CustomAgGrid_v3] set_data rows=%d reason=%r", len(self._rows), reason)

        self.grid.options["rowData"] = self._rows
        self.grid.update()

        ui.timer(self._restore_slow_s, lambda: self._restore_column_state(tag="set_data_slow"), once=True)

    def select_row(self, row_id: RowId, *, ensure_visible: bool = True) -> None:
        """Programmatically select a row by id."""
        # Step 2: queue if grid isn't ready yet
        if not self._grid_ready:
            self._pending_select = (row_id, ensure_visible)
            logger.debug("[CustomAgGrid_v3] queued select_row row_id=%r (grid not ready yet)", row_id)
            return
        asyncio.create_task(self._select_row_by_id_js(row_id=row_id, ensure_visible=ensure_visible))

    def clear_selection(self) -> None:
        """Clear any selected rows."""
        try:
            self.grid.run_grid_method("deselectAll")
        except Exception:
            pass

    def on_selection_changed(self, callback: SelectionCallback | None) -> None:
        """Subscribe to selection changes (row dict or None)."""
        self._selection_cb = callback

    def on_cell_value_changed(self, callback: CellEditCallback | None) -> None:
        """Subscribe to committed cell edits (double-click edit => commit)."""
        self._cell_edit_cb = callback

    def reset_column_state(self) -> None:
        """Forget saved column state (width/sort/etc)."""
        self._saved_column_state = None
        logger.debug("[CustomAgGrid_v3] reset_column_state")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _on_first_data_rendered(self, e: Any) -> None:
        """Mark grid ready, restore column state, and apply any queued selection."""
        self._grid_ready = True
        self._restore_column_state(tag="firstDataRendered")

        if self._pending_select is not None:
            row_id, ensure_visible = self._pending_select
            self._pending_select = None
            logger.debug("[CustomAgGrid_v3] applying queued select_row row_id=%r", row_id)
            try:
                await self._select_row_by_id_js(row_id=row_id, ensure_visible=ensure_visible)
            except Exception:
                pass

    def _schedule_capture_state(self, why: str) -> None:
        if self._capture_scheduled:
            return
        self._last_capture_reason = why
        self._capture_scheduled = True

        def _kickoff() -> None:
            self._capture_scheduled = False
            asyncio.create_task(self._capture_column_state_js(reason=self._last_capture_reason))

        ui.timer(self._capture_debounce_s, _kickoff, once=True)

    async def _capture_column_state_js(self, *, reason: str) -> None:
        js = js_capture_column_state(self.grid.id)
        try:
            result = await self.grid.client.run_javascript(js)
        except Exception:
            return

        if isinstance(result, list):
            self._saved_column_state = result
            logger.debug("[CustomAgGrid_v3] saved column state entries=%d (%s)", len(result), reason)

            if reason in {"columnResized", "sortChanged"}:
                try:
                    await self.grid.client.run_javascript(js_focus_selected_row_first_col(self.grid.id))
                except Exception:
                    pass

    def _restore_column_state(self, *, tag: str) -> None:
        if not self._saved_column_state:
            return
        self.grid.run_grid_method("applyColumnState", {"state": self._saved_column_state, "applyOrder": True})

    async def _on_selection_changed_event(self, e: Any) -> None:
        if not self._selection_cb:
            return
        try:
            rows = await self.grid.run_grid_method("getSelectedRows")
        except Exception:
            self._selection_cb(None)
            return

        if isinstance(rows, list) and rows:
            self._selection_cb(rows[0])
        else:
            self._selection_cb(None)

    def _try_update_local_rows(self, *, row_id: str, field: str, new_value: Any) -> None:
        """Best-effort: keep self._rows roughly in sync with edits (no set_data call)."""
        for r in self._rows:
            if str(r.get(self._row_id_field)) == row_id:
                r[field] = new_value
                return

    def _extract_row_id_from_event(self, payload: dict[str, Any]) -> Optional[str]:
        data = payload.get("data")
        if isinstance(data, Mapping):
            v = data.get(self._row_id_field)
            if v is not None:
                return str(v)
        node = payload.get("node")
        if isinstance(node, Mapping):
            v = node.get("id")
            if v is not None:
                return str(v)
        return None

    def _extract_field_from_event(self, payload: dict[str, Any]) -> Optional[str]:
        coldef = payload.get("colDef")
        if isinstance(coldef, Mapping):
            f = coldef.get("field")
            if isinstance(f, str) and f:
                return f
        column = payload.get("column")
        if isinstance(column, Mapping):
            f = column.get("colId")
            if isinstance(f, str) and f:
                return f
        col_id = payload.get("colId")
        if isinstance(col_id, str) and col_id:
            return col_id
        return None

    def _on_cell_value_changed_event(self, e: Any) -> None:
        payload = getattr(e, "args", None)
        if not isinstance(payload, dict):
            return

        row_id = self._extract_row_id_from_event(payload)
        field = self._extract_field_from_event(payload)
        if not row_id or not field:
            return

        old_value = payload.get("oldValue")
        new_value = payload.get("newValue")

        data = payload.get("data")
        row: RowDict = dict(data) if isinstance(data, Mapping) else {}

        self._try_update_local_rows(row_id=row_id, field=field, new_value=new_value)

        if self._cell_edit_cb:
            try:
                self._cell_edit_cb(
                    CellEditEvent(
                        row_id=row_id,
                        field=field,
                        old_value=old_value,
                        new_value=new_value,
                        row=row,
                    )
                )
            except Exception as ex:
                logger.debug("[CustomAgGrid_v3] cell_edit_cb failed: %r", ex)

    async def _select_row_by_id_js(self, *, row_id: RowId, ensure_visible: bool) -> None:
        js = js_select_row_by_id(self.grid.id, row_id=row_id, ensure_visible=ensure_visible)
        await self.grid.client.run_javascript(js)

    def _build_column_defs(self) -> list[dict[str, Any]]:
        defs: list[dict[str, Any]] = []
        for c in self._columns:
            d: dict[str, Any] = {
                "field": c.field,
                "headerName": c.header or c.field,
                "sortable": bool(c.sortable),
                "resizable": bool(c.resizable),
                "filter": bool(c.filterable),
                "editable": bool(c.editable),
            }

            if c.editable:
                if c.editor == "text":
                    d["cellEditor"] = "agTextCellEditor"
                elif c.editor == "select":
                    d["cellEditor"] = "agSelectCellEditor"
                    choices = list(c.choices) if c.choices is not None else []
                    d["cellEditorParams"] = {"values": [str(x) for x in choices]}

            if c.extra_grid_options:
                d.update(dict(c.extra_grid_options))

            defs.append(d)
        return defs

    def _ensure_rows(self, data: DataLike) -> RowsLike:
        if isinstance(data, list) and all(isinstance(r, Mapping) for r in data):
            return [dict(r) for r in data]
        raise TypeError("Expected list[dict]")