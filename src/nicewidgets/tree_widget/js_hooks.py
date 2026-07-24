"""JS callback strings injected into AG Grid options for ``TreeWidget``.

These are duplicates of the corresponding hooks in
``nicewidgets.table_widget.js_hooks`` (just the two we use: row clicks and
ArrowUp/ArrowDown row nav). Duplication keeps ``tree_widget`` self-contained
and avoids a soft dependency on the ``table_widget`` package.
"""

from __future__ import annotations


def js_on_row_clicked(*, emit_event: str, row_id_field: str) -> str:
    """Build AG Grid ``onRowClicked`` callback emitting normalized selection payload.

    Args:
        emit_event: NiceGUI ``ui.on`` event name to emit.
        row_id_field: Row dict key whose value is the unique row id.

    Returns:
        JS arrow-function source as a string for assignment to
        ``:onRowClicked`` in AG Grid options.
    """
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
    console.warn('[tree_widget] onRowClicked failed', err);
  }}
}}
""".strip()


def js_on_cell_key_down_select_prev_next(*, emit_event: str, row_id_field: str) -> str:
    """Build ``onCellKeyDown`` callback selecting prev/next displayed row on arrows.

    The AG Grid Enterprise tree view exposes ``getDisplayedRowAtIndex`` /
    ``getDisplayedRowCount`` for the post-filter, post-sort row sequence,
    same as the flat-table case, so this hook works without tree-specific
    branching.

    Args:
        emit_event: NiceGUI ``ui.on`` event name to emit.
        row_id_field: Row dict key whose value is the unique row id.

    Returns:
        JS arrow-function source as a string for assignment to
        ``:onCellKeyDown`` in AG Grid options.
    """
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
    console.warn('[tree_widget] onCellKeyDown failed', err);
  }}
}}
""".strip()


def js_on_row_group_opened(*, emit_event: str, row_id_field: str) -> str:
    """Build AG Grid ``onRowGroupOpened`` callback for tree expansion tracking.

    Args:
        emit_event: NiceGUI ``ui.on`` event name to emit.
        row_id_field: Row dict key whose value is the unique row id.

    Returns:
        JS arrow-function source for ``:onRowGroupOpened``.
    """
    return f"""
(params) => {{
  try {{
    const data = params?.data ?? null;
    const rowId = data ? String(data['{row_id_field}']) : null;
    if (!rowId) return;
    emitEvent('{emit_event}', {{
      rowId: rowId,
      expanded: true,
    }});
  }} catch (err) {{
    console.warn('[tree_widget] onRowGroupOpened failed', err);
  }}
}}
""".strip()


def js_on_row_group_closed(*, emit_event: str, row_id_field: str) -> str:
    """Build AG Grid ``onRowGroupClosed`` callback for tree expansion tracking.

    Args:
        emit_event: NiceGUI ``ui.on`` event name to emit.
        row_id_field: Row dict key whose value is the unique row id.

    Returns:
        JS arrow-function source for ``:onRowGroupClosed``.
    """
    return f"""
(params) => {{
  try {{
    const data = params?.data ?? null;
    const rowId = data ? String(data['{row_id_field}']) : null;
    if (!rowId) return;
    emitEvent('{emit_event}', {{
      rowId: rowId,
      expanded: false,
    }});
  }} catch (err) {{
    console.warn('[tree_widget] onRowGroupClosed failed', err);
  }}
}}
""".strip()
