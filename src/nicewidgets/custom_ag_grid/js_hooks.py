# src/nicewidgets/custom_ag_grid/js_hooks.py
from __future__ import annotations

from typing import Optional


def js_on_row_clicked(*, emit_event: str, row_id_field: str) -> str:
    """Return an AG Grid `onRowClicked(params)` hook that emits a JSON-safe selection event."""
    return f"""
(params) => {{
  try {{
    const rowIndex = params?.rowIndex ?? null;
    const data = params?.data ?? null;
    const rowId = data ? String(data['{row_id_field}']) : null;

    emitEvent('{emit_event}', {{
      source: 'click',
      key: null,
      rowIndex: rowIndex,
      rowId: rowId,
      data: data,
    }});
  }} catch (err) {{
    console.warn('[ag] onRowClicked failed', err);
  }}
}}
""".strip()


def js_on_cell_key_down_select_prev_next(
    *,
    emit_event: str,
    row_id_field: str,
    arrow_up: bool = True,
    arrow_down: bool = True,
) -> str:
    """Return an AG Grid `onCellKeyDown(params)` hook that selects prev/next row on ArrowUp/Down."""
    keys = []
    if arrow_up:
        keys.append("ArrowUp")
    if arrow_down:
        keys.append("ArrowDown")
    key_check = " && ".join([f"key !== '{k}'" for k in keys]) or "false"

    return f"""
(params) => {{
  try {{
    const key = params?.event?.key ?? null;
    const rowIndex = params?.rowIndex ?? null;
    const api = params?.api;

    if ({key_check}) return;
    if (!api || rowIndex === null) return;

    // Prevent AG Grid's default focus-only navigation.
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

    // Keep target row visible and make it the single selected row.
    if (api.ensureIndexVisible) api.ensureIndexVisible(target);
    if (api.deselectAll) api.deselectAll();

    const node = api.getDisplayedRowAtIndex ? api.getDisplayedRowAtIndex(target) : null;
    if (node && node.setSelected) {{
      node.setSelected(true, true);
    }}

    // Keep focus aligned with selection (helps repeated arrows).
    try {{
      const colId = params?.column?.getId ? params.column.getId() : null;
      if (colId && api.setFocusedCell) api.setFocusedCell(target, colId);
    }} catch (e) {{}}

    const data = node?.data ?? null;
    const rowId = data ? String(data['{row_id_field}']) : null;

    emitEvent('{emit_event}', {{
      source: 'keydown',
      key: key,
      rowIndex: target,
      rowId: rowId,
      data: data,
    }});
  }} catch (err) {{
    console.warn('[ag] onCellKeyDown failed', err);
  }}
}}
""".strip()


def js_on_cell_double_clicked_start_editing() -> str:
    """Return an AG Grid `onCellDoubleClicked(params)` hook that starts editing the clicked cell."""
    return """
(params) => {
  try {
    const api = params?.api;
    const colId = params?.column?.getId ? params.column.getId() : null;
    const rowIndex = params?.rowIndex ?? null;
    if (!api || colId === null || rowIndex === null) return;

    api.startEditingCell({
      rowIndex: rowIndex,
      colKey: colId,
    });
  } catch (err) {
    console.warn('[ag] onCellDoubleClicked failed', err);
  }
}
""".strip()


def js_on_cell_editing_stopped_emit_change(
    *,
    emit_event: str,
    row_id_field: str,
    include_row_data: bool = True,
) -> str:
    """Return an AG Grid `onCellEditingStopped(params)` hook that emits edit-finished event if changed."""
    include_data = "true" if include_row_data else "false"
    return f"""
(params) => {{
  try {{
    const data = params?.data ?? null;
    const colId = params?.column?.getId?.() ?? null;
    const rowIndex = params?.rowIndex ?? null;

    const oldValue = params?.oldValue;
    const newValue = params?.value;

    if (oldValue === newValue) return;

    const rowId = data ? String(data['{row_id_field}']) : null;

    emitEvent('{emit_event}', {{
      rowIndex: rowIndex,
      rowId: rowId,
      colId: colId,
      oldValue: oldValue,
      newValue: newValue,
      data: { "data" if include_data else "null" },
    }});
  }} catch (err) {{
    console.warn('[ag] onCellEditingStopped failed', err);
  }}
}}
""".strip()