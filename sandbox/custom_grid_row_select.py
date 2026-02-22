# sandbox/aggrid_edit_on_dblclick_poc.py
"""
NiceGUI + AG Grid POC (12 rows):

Features:
- Single row selection (mouse click + ArrowUp/ArrowDown)
- Dedupe: one Python "SELECT_CHANGE" event per actual selection change
- Hide focused-cell outline via CSS (WITHOUT breaking keyboard nav)
- Cell editing on double-click
- Emit one Python "CELL_EDIT" event when editing finishes (only if value changed)

Run:
    uv run python sandbox/aggrid_edit_on_dblclick_poc.py
or:
    python sandbox/aggrid_edit_on_dblclick_poc.py
"""

from __future__ import annotations

from typing import Any, Optional

from nicegui import events, ui

from nicewidgets.utils.logging import configure_logging


class AgGridEditOnDblClickPOC:
    """Minimal wrapper: row select + keyboard navigation + dblclick edit + edit-finished events."""

    def __init__(self, rows: list[dict[str, Any]], *, row_id_field: str = "id") -> None:
        self._rows = rows
        self._row_id_field = row_id_field

        # Unified selection-change event emitted from JS (deduped in Python)
        self._select_event_name = "aggrid_select_change"
        # Cell edit finished event emitted from JS
        self._edit_event_name = "aggrid_cell_edit_finished"

        self._last_emitted_row_id: Optional[str] = None

        self._inject_css_hide_cell_focus()

        self.grid = ui.aggrid(self._grid_options()).classes("w-full h-96")

        # Optional debug: you can remove later; these may fire twice (deselect + select)
        self.grid.on("cellClicked", self.on_cell_clicked)
        self.grid.on("rowSelected", self.on_row_selected)

        # Unified, deduped signals
        ui.on(self._select_event_name, self.on_select_change_emitted)
        ui.on(self._edit_event_name, self.on_cell_edit_finished_emitted)

        ui.label("Click a row to select. Double-click a cell to edit. ArrowUp/ArrowDown selects prev/next row.").classes(
            "text-sm text-gray-600"
        )

    # --------------------------
    # Page CSS
    # --------------------------

    def _inject_css_hide_cell_focus(self) -> None:
        """Hide focused-cell visuals without disabling focus (keeps keyboard navigation working)."""
        ui.add_head_html(
            """
<style>
/* Hide AG Grid's focused-cell outline/border without disabling focus (keeps keyboard nav working) */
.ag-root .ag-cell:focus {
  outline: none !important;
}

/* Different themes/versions use these classes for the focus ring */
.ag-root .ag-cell-focus,
.ag-root .ag-cell-focus:not(.ag-cell-range-selected),
.ag-root .ag-cell-focus:not(.ag-cell-range-selected):focus-within {
  outline: none !important;
  border: none !important;
  box-shadow: none !important;
}

.ag-root .ag-cell-focus.ag-cell {
  outline: none !important;
  box-shadow: none !important;
}
</style>
"""
        )

    # --------------------------
    # Grid options
    # --------------------------

    def _grid_options(self) -> dict[str, Any]:
        keys: list[str] = []
        seen: set[str] = set()
        for r in self._rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)

        if self._row_id_field in keys:
            keys.remove(self._row_id_field)
            keys.insert(0, self._row_id_field)

        col_defs = [{"headerName": k, "field": k} for k in keys]

        # AG Grid v32+ prefers object form
        row_selection_obj = {
            "mode": "singleRow",
            "enableClickSelection": True,
            "checkboxes": False,
        }

        # Emit selection-change on mouse click (one event)
        js_on_row_clicked = f"""
(params) => {{
  try {{
    const rowIndex = params?.rowIndex ?? null;
    const data = params?.data ?? null;
    const idField = '{self._row_id_field}';
    const rowId = data ? String(data[idField]) : null;

    console.log('[ag] onRowClicked', {{ rowIndex, rowId }});

    emitEvent('{self._select_event_name}', {{
      source: 'click',
      key: null,
      rowIndex: rowIndex,
      rowId: rowId,
    }});
  }} catch (err) {{
    console.warn('[ag] onRowClicked failed', err);
  }}
}}
"""

        # ArrowUp/Down selects prev/next displayed row and emits selection-change (one event)
        js_on_cell_key_down = f"""
(params) => {{
  try {{
    const key = params?.event?.key ?? null;
    const rowIndex = params?.rowIndex ?? null;
    const api = params?.api;

    if (key !== 'ArrowUp' && key !== 'ArrowDown') return;
    if (!api || rowIndex === null) return;

    console.log('[ag] onCellKeyDown', {{ key, rowIndex }});

    try {{
      params.event.preventDefault();
      params.event.stopPropagation();
    }} catch (e) {{}}

    const total = api.getDisplayedRowCount();
    const current = Number(rowIndex);
    if (!Number.isFinite(current)) return;

    const target = (key === 'ArrowUp')
      ? Math.max(0, current - 1)
      : Math.min(total - 1, current + 1);

    api.ensureIndexVisible(target);
    api.deselectAll();

    const node = api.getDisplayedRowAtIndex(target);
    if (node && node.setSelected) {{
      node.setSelected(true, true);
    }}

    const colId = params?.column?.getId ? params.column.getId() : null;
    if (colId && api.setFocusedCell) {{
      api.setFocusedCell(target, colId);
    }}

    const idField = '{self._row_id_field}';
    const rowId = node?.data ? String(node.data[idField]) : null;

    emitEvent('{self._select_event_name}', {{
      source: 'keydown',
      key: key,
      rowIndex: target,
      rowId: rowId,
    }});
  }} catch (err) {{
    console.warn('[ag] onCellKeyDown failed', err);
  }}
}}
"""

        # Start editing on double-click (forces consistent behavior across grid settings)
        js_on_cell_double_clicked = """
(params) => {
  try {
    const api = params?.api;
    const colId = params?.column?.getId ? params.column.getId() : null;
    const rowIndex = params?.rowIndex ?? null;

    console.log('[ag] onCellDoubleClicked', { rowIndex, colId });

    if (!api || colId === null || rowIndex === null) return;

    api.startEditingCell({
      rowIndex: rowIndex,
      colKey: colId,
    });
  } catch (err) {
    console.warn('[ag] onCellDoubleClicked failed', err);
  }
}
"""

        # Emit edit-finished event (only if value changed)
        js_on_cell_editing_stopped = f"""
(params) => {{
  try {{
    const data = params?.data ?? null;
    const colId = params?.column?.getId?.() ?? null;
    const rowIndex = params?.rowIndex ?? null;

    const oldValue = params?.oldValue;
    const newValue = params?.value;

    console.log('[ag] onCellEditingStopped', {{ rowIndex, colId, oldValue, newValue }});

    // Only report real changes
    if (oldValue === newValue) return;

    const idField = '{self._row_id_field}';
    const rowId = data ? String(data[idField]) : null;

    emitEvent('{self._edit_event_name}', {{
      rowIndex: rowIndex,
      rowId: rowId,
      colId: colId,
      oldValue: oldValue,
      newValue: newValue,
    }});
  }} catch (err) {{
    console.warn('[ag] onCellEditingStopped failed', err);
  }}
}}
"""

        return {
            "columnDefs": col_defs,
            "rowData": self._rows,

            # Make cells editable (for the POC we keep it simple: everything editable)
            "defaultColDef": {
                "editable": True,
                "resizable": True,
                "sortable": True,
                "filter": True,
            },

            "rowSelection": row_selection_obj,
            "rowHeight": 28,
            "headerHeight": 30,

            # Stable row id (NiceGUI run_row_method uses this)
            ":getRowId": f"(p) => p.data && String(p.data['{self._row_id_field}'])",

            # Hooks implemented inside AG Grid (crucial: full params, including DOM key event)
            ":onRowClicked": js_on_row_clicked,
            ":onCellKeyDown": js_on_cell_key_down,
            ":onCellDoubleClicked": js_on_cell_double_clicked,
            ":onCellEditingStopped": js_on_cell_editing_stopped,
        }

    # --------------------------
    # Python debug handlers (optional)
    # --------------------------

    def on_cell_clicked(self, e: events.GenericEventArguments) -> None:
        args = e.args or {}
        data = args.get("data") or {}
        print(f"[py] cellClicked rowIndex={args.get('rowIndex')} rowId={data.get(self._row_id_field)}")

    def on_row_selected(self, e: events.GenericEventArguments) -> None:
        args = e.args or {}
        data = args.get("data") or {}
        print(
            f"[py] rowSelected selected={args.get('selected')} rowIndex={args.get('rowIndex')} rowId={data.get(self._row_id_field)}"
        )

    # --------------------------
    # Python: deduped selection-change callback
    # --------------------------

    def on_select_change_emitted(self, e: events.GenericEventArguments) -> None:
        args = e.args or {}
        row_id = args.get("rowId")
        row_index = args.get("rowIndex")
        source = args.get("source")
        key = args.get("key")

        if row_id is None:
            return

        # Dedup: only emit when rowId changes
        if row_id == self._last_emitted_row_id:
            return
        self._last_emitted_row_id = row_id

        print(f"[py] SELECT_CHANGE source={source} key={key} rowIndex={row_index} rowId={row_id}")

    # --------------------------
    # Python: cell edit finished callback
    # --------------------------

    def on_cell_edit_finished_emitted(self, e: events.GenericEventArguments) -> None:
        args = e.args or {}
        print(
            "[py] CELL_EDIT "
            f"rowId={args.get('rowId')} "
            f"rowIndex={args.get('rowIndex')} "
            f"colId={args.get('colId')} "
            f"{args.get('oldValue')} -> {args.get('newValue')}"
        )


def demo_rows_12() -> list[dict[str, Any]]:
    return [{"id": f"row_{i:03d}", "name": f"Item {i}", "value": i * 10} for i in range(1, 13)]


@ui.page("/")
def index() -> None:
    print("[py] page loaded")
    ui.label("AG Grid: dbl-click edit + edit-finished events (POC, 12 rows)").classes("text-lg font-semibold")
    AgGridEditOnDblClickPOC(demo_rows_12(), row_id_field="id")


if __name__ in {"__main__", "__mp_main__"}:
    configure_logging(level="INFO")
    ui.run(reload=False)