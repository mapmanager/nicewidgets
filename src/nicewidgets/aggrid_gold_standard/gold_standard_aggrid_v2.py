"""Gold standard aggrid v2 for NiceGUI with right-click column toggle and in-cell select editing.

- Context menu: lists columns with ✓, toggle visibility, Show all / Hide all
- Editable columns: double-click shows AG Grid agSelectCellEditor (in-cell dropdown)
  Options from unique values in rowData. Note: agSelectCellEditor does not support
  typing new values; use editable_columns for preset-only selection.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import pandas as pd
from nicegui import ui
from nicegui.events import GenericEventArguments

from nicewidgets.custom_ag_grid.js_hooks import js_on_cell_double_clicked_start_editing
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)

# Columns to exclude from the toggle menu (e.g. row index columns)
CONTEXT_MENU_EXCLUDED_COLUMNS = frozenset({"__row_index__"})
CHECK_EMOJI = "✓"


def _unique_from_row_data(row_data: list[dict[str, Any]], col: str) -> list[str]:
    """Return sorted unique string values for a column from row_data (list of dicts)."""
    if not row_data or not col:
        return []
    seen: set[str] = set()
    for row in row_data:
        if col in row and row[col] is not None:
            seen.add(str(row[col]))
    return sorted(seen)


def _js_get_context_menu_column_toggle(
    *,
    column_fields: list[str],
    excluded: frozenset[str] = CONTEXT_MENU_EXCLUDED_COLUMNS,
    check_emoji: str = CHECK_EMOJI,
) -> str:
    """Return AG Grid getContextMenuItems callback as a JS function string.

    Builds a context menu with:
    - One item per column (from column_fields, excluding excluded)
    - Check emoji prefix for visible columns
    - Toggle visibility on click
    - Separator
    - "Show all" and "Hide all" actions
    """
    fields_json = json.dumps(column_fields)
    excl_json = json.dumps(list(excluded))
    return f"""
(params) => {{
  try {{
    const api = params?.api;
    if (!api) return [];

    const COLUMN_FIELDS = {fields_json};
    const EXCLUDED = new Set({excl_json});
    const CHECK = "{check_emoji}";

    const toggleableFields = COLUMN_FIELDS.filter(f => !EXCLUDED.has(f));

    const items = [];

    const columns = api.getColumns ? api.getColumns() : [];
    const colById = new Map();
    if (columns) {{
      for (const col of columns) {{
        const id = (typeof col.getColId === "function" ? col.getColId() : col.colId) ?? "";
        colById.set(id, col);
      }}
    }}

    const isColVisible = (col) => typeof col.isVisible === "function" ? col.isVisible() : (col.visible !== false);
    const setColVisible = (col, vis) => {{
      if (typeof col.setVisible === "function") col.setVisible(vis);
    }};

    for (const field of toggleableFields) {{
      const col = colById.get(field) || columns?.find(c => (typeof c.getColId === "function" ? c.getColId() : c.colId) === field);
      if (!col) continue;

      const visible = isColVisible(col);
      const headerName = col.colDef?.headerName ?? col.colDef?.field ?? field;

      items.push({{
        name: (visible ? CHECK + " " : "") + headerName,
        checked: visible,
        action: () => {{
          if (api.setColumnsVisible) {{
            api.setColumnsVisible([field], !visible);
          }} else {{
            setColVisible(col, !visible);
          }}
        }}
      }});
    }}

    if (items.length > 0) {{
      items.push("separator");
      items.push({{
        name: "Show all",
        action: () => {{
          const cols = api.getColumns ? api.getColumns() : [];
          if (cols) {{
            const ids = cols.map(c => typeof c.getColId === "function" ? c.getColId() : c.colId).filter(Boolean);
            if (ids.length && api.setColumnsVisible) api.setColumnsVisible(ids, true);
            else cols.forEach(c => setColVisible(c, true));
          }}
        }}
      }});
      items.push({{
        name: "Hide all",
        action: () => {{
          const cols = api.getColumns ? api.getColumns() : [];
          if (cols) {{
            const ids = cols.map(c => typeof c.getColId === "function" ? c.getColId() : c.colId).filter(Boolean);
            if (ids.length && api.setColumnsVisible) api.setColumnsVisible(ids, false);
            else cols.forEach(c => setColVisible(c, false));
          }}
        }}
      }});
    }}

    return items;
  }} catch (err) {{
    console.warn("[gold_standard_aggrid_v2] getContextMenuItems failed", err);
    return [];
  }}
}}
""".strip()


def gold_standard_aggrid_v2(
    df: pd.DataFrame,
    *,
    unique_row_id_col: Optional[str] = None,
    row_select_callback: Optional[Callable[[Any, dict[str, Any]], None]] = None,
    row_dblclick_callback: Optional[Callable[[Any, str, dict[str, Any]], None]] = None,
    editable_columns: Optional[list[str]] = None,
    enable_column_toggle_context_menu: bool = True,
    use_ui_context_menu: bool = True,
    columns: Optional[list[str]] = None,
) -> ui.aggrid:
    """Gold standard AG Grid with right-click column toggle and in-cell select editing.

    Same core API as gold_standard_aggrid, plus:
    - row_dblclick_callback: (row_id, column_field, row_dict) -> None; called for non-editable columns
    - editable_columns: columns that use in-cell agSelectCellEditor on double-click (dropdown in cell)
    - enable_column_toggle_context_menu: show right-click menu to toggle columns (default True)
    - use_ui_context_menu: use NiceGUI ui.context_menu() (works reliably when AG Grid native fails)
    - columns: optional subset of df.columns to display; if None, uses all df.columns

    Editable columns: double-click shows in-cell agSelectCellEditor (dropdown in the cell).
    Options from unique values in rowData. Requires unique_row_id_col for getRowId.
    """
    display_columns = columns if columns is not None else list(df.columns)
    toggleable = [c for c in display_columns if c not in CONTEXT_MENU_EXCLUDED_COLUMNS]
    editable_set = frozenset(editable_columns) if editable_columns else frozenset()

    def _on_row_selected(e: GenericEventArguments) -> None:
        _row_dict = e.args.get("data") or {}
        _unique_id = _row_dict.get(unique_row_id_col) if unique_row_id_col else None
        if row_select_callback is not None:
            row_select_callback(_unique_id, _row_dict)

    def _on_cell_dblclick(e: GenericEventArguments) -> None:
        if row_dblclick_callback is None:
            return
        row_dict = e.args.get("data") or {}
        col_def = e.args.get("colDef") or {}
        col_field = col_def.get("field") or e.args.get("colId") or ""
        row_id = row_dict.get(unique_row_id_col) if unique_row_id_col else None
        if col_field in editable_set:
            return
        row_dblclick_callback(row_id, col_field, row_dict)

    with ui.column().classes("w-full h-full min-h-0") as container:
        aggrid = ui.aggrid.from_pandas(df).classes("w-full aggrid-compact")

        # NiceGUI ui.context_menu() - parallel to AG Grid native; attaches to container (kymflow pattern)
        if enable_column_toggle_context_menu and use_ui_context_menu and toggleable:
            _visible_cache: dict[str, bool] = {c: True for c in toggleable}

            def _toggle_col(col: str) -> None:
                _visible_cache[col] = not _visible_cache[col]
                aggrid.run_grid_method("setColumnsVisible", [col], _visible_cache[col])

            def _show_all() -> None:
                for c in toggleable:
                    _visible_cache[c] = True
                aggrid.run_grid_method("setColumnsVisible", toggleable, True)

            def _hide_all() -> None:
                for c in toggleable:
                    _visible_cache[c] = False
                aggrid.run_grid_method("setColumnsVisible", toggleable, False)

            def _rebuild_menu() -> None:
                with ctx_menu.clear():
                    for col in toggleable:
                        label = f"{CHECK_EMOJI} {col}" if _visible_cache[col] else f"   {col}"
                        ui.menu_item(label, on_click=lambda c=col: _toggle_col(c))
                    ui.separator()
                    ui.menu_item("Show all", on_click=_show_all)
                    ui.menu_item("Hide all", on_click=_hide_all)

            with ui.context_menu() as ctx_menu:
                pass  # populated dynamically on contextmenu

            container.on("contextmenu", _rebuild_menu)
            _rebuild_menu()  # initial build so menu has content on first open

        row_data = aggrid.options.get("rowData") or df.to_dict("records")
        column_defs = []
        for c in display_columns:
            col_def: dict[str, Any] = {
                "headerName": c,
                "field": c,
                "editable": c in editable_set,
                "checkboxSelection": False,
                "headerCheckboxSelection": False,
                "sortable": True,
                "resizable": True,
            }
            if c in editable_set:
                values = _unique_from_row_data(row_data, c)
                col_def[":cellEditor"] = "'agSelectCellEditor'"
                col_def["cellEditorParams"] = {"values": values}
            column_defs.append(col_def)

        aggrid.options["columnDefs"] = column_defs
        aggrid.options["rowSelection"] = "single"
        aggrid.options["suppressRowClickSelection"] = False
        aggrid.options["stopEditingWhenCellsLoseFocus"] = True

        if editable_set:
            aggrid.options[":onCellDoubleClicked"] = js_on_cell_double_clicked_start_editing()

        if unique_row_id_col:
            aggrid.options[":getRowId"] = (
                f"(params) => params.data && String(params.data['{unique_row_id_col}'])"
            )

        # AG Grid native getContextMenuItems (first attempt; may not work with downgraded AG Grid)
        if enable_column_toggle_context_menu:
            aggrid.options[":getContextMenuItems"] = _js_get_context_menu_column_toggle(
                column_fields=toggleable,
                excluded=CONTEXT_MENU_EXCLUDED_COLUMNS,
                check_emoji=CHECK_EMOJI,
            )

        aggrid.update()

        # logger.info('row data is:')
        # print(aggrid.options["rowData"])

    if row_select_callback is not None:
        aggrid.on("rowSelected", _on_row_selected)

    if row_dblclick_callback is not None:
        aggrid.on("cellDoubleClicked", _on_cell_dblclick)

    return aggrid
