from __future__ import annotations


def js_on_row_clicked(*, emit_event: str, row_id_field: str) -> str:
    """Build AG Grid ``onRowClicked`` callback emitting normalized selection payload."""
    return f"""
(params) => {{
  try {{
    const rowIndex = params?.rowIndex ?? null;
    const data = params?.data ?? null;
    const rowId = data ? String(data['{row_id_field}']) : null;
    emitEvent('{emit_event}', {{
      source: 'click',
      rowIndex: rowIndex,
      rowId: rowId,
      data: data,
    }});
  }} catch (err) {{
    console.warn('[table_widget] onRowClicked failed', err);
  }}
}}
""".strip()


def js_on_cell_key_down_select_prev_next(*, emit_event: str, row_id_field: str) -> str:
    """Build ``onCellKeyDown`` callback selecting prev/next displayed row on arrows."""
    return f"""
(params) => {{
  try {{
    const key = params?.event?.key ?? null;
    if (key !== 'ArrowUp' && key !== 'ArrowDown') return;
    const api = params?.api;
    const current = Number(params?.rowIndex ?? -1);
    if (!api || !Number.isFinite(current)) return;
    try {{
      params.event.preventDefault();
      params.event.stopPropagation();
    }} catch (e) {{}}
    const total = api.getDisplayedRowCount();
    if (!Number.isFinite(total) || total <= 0) return;
    const target = (key === 'ArrowUp') ? Math.max(0, current - 1) : Math.min(total - 1, current + 1);
    if (api.ensureIndexVisible) api.ensureIndexVisible(target);
    if (api.deselectAll) api.deselectAll();
    const node = api.getDisplayedRowAtIndex ? api.getDisplayedRowAtIndex(target) : null;
    if (!node || !node.setSelected) return;
    node.setSelected(true, true);
    const data = node.data ?? null;
    const rowId = data ? String(data['{row_id_field}']) : null;
    emitEvent('{emit_event}', {{
      source: 'keydown',
      key: key,
      rowIndex: target,
      rowId: rowId,
      data: data,
    }});
  }} catch (err) {{
    console.warn('[table_widget] onCellKeyDown failed', err);
  }}
}}
""".strip()


def js_on_cell_double_clicked_start_editing() -> str:
    """Build ``onCellDoubleClicked`` callback that starts editing clicked cell."""
    return """
(params) => {
  try {
    const api = params?.api;
    const colId = params?.column?.getId ? params.column.getId() : null;
    const rowIndex = params?.rowIndex ?? null;
    if (!api || colId === null || rowIndex === null) return;
    api.startEditingCell({ rowIndex: rowIndex, colKey: colId });
  } catch (err) {
    console.warn('[table_widget] onCellDoubleClicked failed', err);
  }
}
""".strip()


def js_on_cell_editing_stopped_emit_change(*, emit_event: str, row_id_field: str) -> str:
    """Build ``onCellEditingStopped`` callback emitting changed-value payload."""
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
      data: data,
    }});
  }} catch (err) {{
    console.warn('[table_widget] onCellEditingStopped failed', err);
  }}
}}
""".strip()
