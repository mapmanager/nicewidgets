# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/js_hooks_v3.py
# v3-only JS helpers (keyboard selection, capture state, programmatic selection).
# NOTE: Do not use relative imports elsewhere; keep this a standalone module.

from __future__ import annotations

import json


def js_get_row_id(row_id_field: str) -> str:
    field_js = json.dumps(row_id_field)
    return f"""
(params) => {{
  try {{
    const d = params?.data;
    if (!d) return null;
    const v = d[{field_js}];
    return (v === null || v === undefined) ? null : String(v);
  }} catch (e) {{
    return null;
  }}
}}
""".strip()


def js_capture_column_state(grid_element_id: str) -> str:
    grid_id_js = json.dumps(grid_element_id)
    return f"""
(() => {{
  try {{
    const el = getElement({grid_id_js});
    if (!el) return null;

    const api = el.api;
    if (api && typeof api.getColumnState === 'function') return api.getColumnState();

    const columnApi = el.columnApi;
    if (columnApi && typeof columnApi.getColumnState === 'function') return columnApi.getColumnState();

    const vpc = el.__vueParentComponent;
    const exposed = vpc?.exposed;
    const ctx = vpc?.ctx;

    const api2 = exposed?.api ?? ctx?.api;
    if (api2 && typeof api2.getColumnState === 'function') return api2.getColumnState();

    const columnApi2 = exposed?.columnApi ?? ctx?.columnApi;
    if (columnApi2 && typeof columnApi2.getColumnState === 'function') return columnApi2.getColumnState();

    return null;
  }} catch (e) {{
    return null;
  }}
}})();
""".strip()


def js_on_row_clicked_focus_clicked_col() -> str:
    """Select clicked row and focus the clicked column (no snap-to-first-col)."""
    return """
(params) => {
  try {
    const api = params?.api;
    const rowIndex = params?.rowIndex ?? null;
    if (!api || rowIndex === null) return;

    const node = params?.node ?? null;
    if (node && node.setSelected) node.setSelected(true, true);

    const colId = params?.column?.getId ? params.column.getId() : null;
    if (colId && api.setFocusedCell) api.setFocusedCell(rowIndex, colId);
  } catch (e) {
    console.warn('[CustomAgGrid_v3] onRowClicked focus failed', e);
  }
}
""".strip()


def js_on_cell_key_down_select_prev_next() -> str:
    """
    ArrowUp/ArrowDown: select prev/next row.

    IMPORTANT: If a cell editor is open (editing), do NOT hijack arrow keys.
    This keeps select editor UX sane.
    """
    return """
(params) => {
  try {
    const api = params?.api;
    if (!api) return;

    // If editing is active, leave arrow keys to the editor.
    try {
      const editing = api.getEditingCells ? api.getEditingCells() : [];
      if (editing && editing.length > 0) return;
    } catch (e) {}

    const key = params?.event?.key ?? null;
    const rowIndex = params?.rowIndex ?? null;

    if (key !== 'ArrowUp' && key !== 'ArrowDown') return;
    if (rowIndex === null) return;

    try { params.event.preventDefault(); params.event.stopPropagation(); } catch (e) {}

    const total = api.getDisplayedRowCount ? api.getDisplayedRowCount() : 0;
    const current = Number(rowIndex);
    if (!Number.isFinite(current) || total <= 0) return;

    const target = (key === 'ArrowUp')
      ? Math.max(0, current - 1)
      : Math.min(total - 1, current + 1);

    if (api.deselectAll) api.deselectAll();

    const node = api.getDisplayedRowAtIndex ? api.getDisplayedRowAtIndex(target) : null;
    if (node && node.setSelected) node.setSelected(true, true);

    if (api.ensureIndexVisible) api.ensureIndexVisible(target, 'auto');

    // Keep focus in the grid to ensure continuous keynav.
    try {
      const colId = params?.column?.getId ? params.column.getId() : null;
      if (colId && api.setFocusedCell) api.setFocusedCell(target, colId);
    } catch (e) {}

    try { if (api.clearRangeSelection) api.clearRangeSelection(); } catch (e) {}
  } catch (err) {
    console.warn('[CustomAgGrid_v3] onCellKeyDown failed', err);
  }
}
""".strip()


def js_focus_selected_row_first_col(grid_element_id: str) -> str:
    """After resize/sort, header may steal focus. Refocus selected row (first displayed col)."""
    grid_id_js = json.dumps(grid_element_id)
    return f"""
(() => {{
  try {{
    const el = getElement({grid_id_js});
    if (!el) return false;

    let api = el.api;
    const vpc = el.__vueParentComponent;
    api = api ?? vpc?.exposed?.api ?? vpc?.ctx?.api;
    if (!api) return false;

    const nodes = api.getSelectedNodes ? api.getSelectedNodes() : [];
    if (!nodes || !nodes.length) return false;

    const rowIndex = nodes[0].rowIndex;
    if (typeof rowIndex !== 'number') return false;

    const cols = api.getAllDisplayedColumns ? api.getAllDisplayedColumns() : [];
    const colId = cols.length ? (cols[0].getColId ? cols[0].getColId() : null) : null;
    if (colId && api.setFocusedCell) api.setFocusedCell(rowIndex, colId);

    return true;
  }} catch (e) {{
    return false;
  }}
}})();
""".strip()


def js_select_row_by_id(grid_element_id: str, *, row_id: str | int, ensure_visible: bool) -> str:
    """Select a row by id using api.getRowNode(String(id))."""
    grid_id_js = json.dumps(grid_element_id)
    row_id_js = json.dumps(row_id)
    ensure_js = "true" if ensure_visible else "false"
    return f"""
(() => {{
  try {{
    const el = getElement({grid_id_js});
    if (!el) return false;

    let api = el.api;
    const vpc = el.__vueParentComponent;
    api = api ?? vpc?.exposed?.api ?? vpc?.ctx?.api;
    if (!api) return false;

    const id = {row_id_js};
    const node = api.getRowNode?.(String(id));
    if (!node) return false;

    node.setSelected?.(true, true);

    if ({ensure_js}) {{
      const idx = node.rowIndex;
      if (typeof idx === 'number') api.ensureIndexVisible?.(idx, 'auto');
    }}
    return true;
  }} catch (e) {{
    return false;
  }}
}})();
""".strip()