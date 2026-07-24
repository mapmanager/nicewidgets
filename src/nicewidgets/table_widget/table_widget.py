"""Composable NiceGUI ``ui.aggrid`` table with id-keyed mutations and optional JS-enhanced hooks."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nicegui import events, ui

from nicewidgets.table_widget.column_def import ColumnDef
from nicewidgets.table_widget.config import SelectionMode, TableWidgetConfig
from nicewidgets.utils.clipboard import copy_to_clipboard
from nicewidgets.utils.logging import get_logger
from nicewidgets.table_widget.js_hooks import (
    js_on_cell_double_clicked_start_editing,
    js_on_cell_editing_stopped_emit_change,
    js_on_cell_key_down_select_prev_next,
    js_on_row_clicked,
)


logger = get_logger(__name__)


def validate_row_id_field(row_id_field: str) -> None:
    """Validate ``row_id_field`` is usable as a non-empty dict key."""
    if not row_id_field or not str(row_id_field).strip():
        raise ValueError('row_id_field must be a non-empty string')


def validate_rows_for_row_id_field(rows: Sequence[Mapping[str, Any]], row_id_field: str) -> None:
    """Ensure each row has a unique non-empty string id at ``row_id_field``."""
    validate_row_id_field(row_id_field)
    seen: set[str] = set()
    for i, row in enumerate(rows):
        if row_id_field not in row:
            raise ValueError(f'Row {i} is missing required id key {row_id_field!r}')
        raw = row[row_id_field]
        if not isinstance(raw, str):
            raise ValueError(f'Row {i}: value at {row_id_field!r} must be str (got {type(raw).__name__})')
        if not raw:
            raise ValueError(f'Row {i}: value at {row_id_field!r} must be a non-empty string')
        if raw in seen:
            raise ValueError(f'Duplicate row id {raw!r} at {row_id_field!r}')
        seen.add(raw)


def _deep_merge_aggrid_options(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge AG Grid option dicts with one-level nested dict merge."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in {'columnDefs', 'rowData'}:
            out[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            merged_inner = copy.deepcopy(out[k])
            merged_inner.update(v)
            out[k] = merged_inner
        else:
            out[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
    return out


def _get_row_id_js_expression(row_id_field: str) -> str:
    """Build JS arrow function string for AG Grid ``getRowId`` dynamic key."""
    key_literal = json.dumps(row_id_field)
    return f'(params) => String(params.data != null ? params.data[{key_literal}] : "")'


class TableWidget:
    """NiceGUI AG Grid wrapper with modular selection, editing, and context-menu composition."""

    def __init__(
        self,
        columns: Sequence[ColumnDef],
        row_id_field: str,
        rows: Sequence[Mapping[str, Any]] | None = None,
        *,
        on_row_selected: Callable[[dict[str, Any]], None] | None = None,
        on_cell_edited: Callable[[str, str, Any, Any, dict[str, Any]], None] | None = None,
        on_build_context_menu: Callable[[TableWidget], None] | None = None,
        config: TableWidgetConfig | None = None,
        grid_options: Mapping[str, Any] | None = None,
    ) -> None:
        validate_row_id_field(row_id_field)
        self._row_id_field = row_id_field
        self._on_row_selected = on_row_selected
        self._on_cell_edited = on_cell_edited
        self._on_build_context_menu = on_build_context_menu
        self._config = config or TableWidgetConfig()
        self._grid_options_user = dict(grid_options or {})
        self._selection_origin = 'internal'

        self._evt_select = f'table_widget_select_{id(self)}'
        self._evt_edit = f'table_widget_edit_{id(self)}'

        self._index_field: str | None = None
        if self._config.show_index_column:
            idx_f = str(self._config.index_field).strip()
            if not idx_f:
                raise ValueError('index_field must be non-empty when show_index_column is true')
            for c in columns:
                if c.field == idx_f:
                    raise ValueError(
                        f'Column field {idx_f!r} conflicts with TableWidgetConfig.index_field; '
                        'rename the column or set a different index_field'
                    )
            self._index_field = idx_f
            index_col = ColumnDef(
                field=idx_f,
                headerName=str(self._config.index_header),
                extra={
                    'editable': False,
                    'sortable': True,
                    'filter': False,
                    'type': 'numericColumn',
                    'maxWidth': 96,
                },
            )
            built_columns = (index_col, *columns)
        else:
            built_columns = tuple(columns)

        self._column_defs: list[dict[str, Any]] = [c.as_aggrid_column_def() for c in built_columns]
        self._rows: list[dict[str, Any]] = [dict(r) for r in (rows or ())]
        validate_rows_for_row_id_field(self._rows, self._row_id_field)
        self._assign_row_indices()

        self._selected_row_ids: list[str] = []
        self._selected_rows: list[dict[str, Any]] = []
        self._last_selected_row_id: str | None = None

        self._root: ui.column | None = None
        self._grid: ui.aggrid | None = None
        self._context_menu: ui.context_menu | None = None

        ui.on(self._evt_select, self._on_select_emitted)
        ui.on(self._evt_edit, self._on_edit_emitted)

    def _assign_row_indices(self) -> None:
        """Set 1-based row index in row data for the synthetic index column."""
        if self._index_field is None:
            return
        for i, row in enumerate(self._rows):
            row[self._index_field] = i + 1

    def build(self, parent: ui.element | None = None) -> ui.column:
        """Create the wrapper + context menu + AG Grid under ``parent`` (or current slot)."""
        # AG Grid requires a real height from its container; provide a sane default
        # when callers do not pass a sized parent container.
        container = parent if parent is not None else ui.column().classes('w-full').style('height: 24rem;')
        with container:
            self._root = ui.column().classes('w-full h-full min-w-0 min-h-0')
            with self._root:
                self._context_menu = ui.context_menu()
                with self._context_menu:
                    self._build_context_menu_content()
                self._root.on('contextmenu', self._on_context_menu_event)
                self._grid = ui.aggrid(
                    self._build_aggrid_options(),
                    auto_size_columns=self._config.auto_size_columns,
                ).classes('w-full h-full min-w-0 min-h-0').style('height: 100%;')
        return self._root


    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable pointer interaction with the table.

        Args:
            enabled: Desired enabled state.
        """
        enabled = bool(enabled)
        if self._root is not None:
            self._root.enabled = enabled
            if enabled:
                self._root.classes(remove='pointer-events-none opacity-60')
            else:
                self._root.classes(add='pointer-events-none opacity-60')
            self._root.update()
        if self._grid is not None:
            self._grid.enabled = enabled
            self._grid.update()

    def get_rows(self) -> list[dict[str, Any]]:
        """Return a copy of internal row data."""
        return [dict(r) for r in self._rows]

    def get_selected_rows(self) -> list[dict[str, Any]]:
        """Return last known selected rows."""
        return [dict(r) for r in self._selected_rows]

    def get_selected_row_ids(self) -> list[str]:
        """Return last known selected row ids."""
        return list(self._selected_row_ids)

    def set_selected_row_ids(self, row_ids: Sequence[str], *, origin: str = 'external') -> None:
        """Programmatically select rows by row id (selection mode aware)."""
        normalized = [str(rid) for rid in row_ids]
        if self._config.selection_mode == 'none':
            self._selected_row_ids = []
            self._selected_rows = []
            self._last_selected_row_id = None
            return
        if self._config.selection_mode == 'single':
            normalized = normalized[:1]

        row_by_id = {str(row[self._row_id_field]): dict(row) for row in self._rows}
        keep = [rid for rid in normalized if rid in row_by_id]
        self._selected_row_ids = keep
        self._selected_rows = [row_by_id[rid] for rid in keep]
        self._last_selected_row_id = keep[0] if keep else None

        if self._grid is None:
            return

        self._selection_origin = origin
        self._grid.run_grid_method('deselectAll')
        for i, rid in enumerate(keep):
            clear = bool(i == 0 and self._config.selection_mode == 'single')
            self._grid.run_row_method(rid, 'setSelected', True, clear)
        self._selection_origin = 'internal'

    def clear_selection(self) -> None:
        """Clear selected-row tracking and grid selection."""
        self._selected_row_ids = []
        self._selected_rows = []
        self._last_selected_row_id = None
        if self._grid is not None:
            self._grid.run_grid_method('deselectAll')

    def set_data(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Replace all rows and refresh the grid."""
        new_rows = [dict(r) for r in rows]
        validate_rows_for_row_id_field(new_rows, self._row_id_field)
        self._rows = new_rows
        self._assign_row_indices()
        self._push_row_data_to_grid()
        if self._config.clear_selection_on_set_data:
            self.clear_selection()

    def upsert_row(self, row: Mapping[str, Any]) -> None:
        """Insert or replace one row by ``row_id_field``."""
        validate_rows_for_row_id_field([row], self._row_id_field)
        rid = str(row[self._row_id_field])
        replacement = dict(row)
        for i, existing in enumerate(self._rows):
            if existing.get(self._row_id_field) == rid:
                self._rows[i] = replacement
                break
        else:
            self._rows.append(replacement)
        self._assign_row_indices()
        self._push_row_data_to_grid()

    def update_row(self, row_id: str, row: Mapping[str, Any]) -> None:
        """Update a single row by id, patching the row node when possible."""
        validate_rows_for_row_id_field([row], self._row_id_field)
        rid = str(row_id)
        replacement = dict(row)
        idx: int | None = None
        for i, existing in enumerate(self._rows):
            if str(existing.get(self._row_id_field)) == rid:
                idx = i
                break
        if idx is None:
            raise ValueError(f'No row with id {rid!r}')
        self._rows[idx] = replacement
        self._assign_row_indices()
        if self._grid is None:
            return
        try:
            self._grid.run_row_method(rid, 'setData', dict(self._rows[idx]))
        except RuntimeError:
            self._push_row_data_to_grid()

    def remove_row(self, row_id: str) -> None:
        """Remove row matching ``row_id``."""
        if not isinstance(row_id, str):
            raise ValueError('remove_row expects row_id: str')
        before = len(self._rows)
        self._rows = [r for r in self._rows if r.get(self._row_id_field) != row_id]
        if len(self._rows) == before:
            raise ValueError(f'No row with id {row_id!r}')
        self._assign_row_indices()
        self._push_row_data_to_grid()
        if row_id in self._selected_row_ids:
            self._selected_row_ids = [rid for rid in self._selected_row_ids if rid != row_id]
            self._selected_rows = [r for r in self._selected_rows if str(r.get(self._row_id_field)) != row_id]
            self._last_selected_row_id = self._selected_row_ids[0] if self._selected_row_ids else None

    def set_column_visible(self, field: str, visible: bool) -> None:
        """Show or hide one column by field."""
        col = self._find_column_def(field)
        col['hide'] = not visible
        self._apply_column_defs_to_grid()

    def toggle_column_visible(self, field: str) -> None:
        """Toggle one column by field."""
        vis = self.get_column_visibility()[field]
        self.set_column_visible(field, not vis)

    def get_column_visibility(self) -> dict[str, bool]:
        """Return ``field -> visible`` map."""
        return {str(c['field']): not bool(c.get('hide', False)) for c in self._column_defs}

    def get_table_as_text(self) -> str:
        """Return current Python-side table rows as tab-separated text.

        The exported data uses the widget's internal row list and current column
        visibility settings. It does not query browser-side AG Grid sort or
        filter state. Column order follows the current ``columnDefs`` order,
        and hidden columns are omitted.

        Returns:
            TSV-formatted text containing one header row and one row per table
            row. Returns an empty string when there are no rows. Tabs and
            newlines inside headers or cell values are normalized to spaces.
        """
        return self._rows_to_tsv(self._rows)

    async def get_displayed_rows(self) -> list[dict[str, Any]]:
        """Return AG Grid rows after browser-side filtering and sorting.

        AG Grid owns sort and filter state in the browser. This method uses
        NiceGUI's ``ui.run_javascript`` escape hatch to query the AG Grid API
        directly with ``forEachNodeAfterFilterAndSort``. If the grid has not
        been built yet, the method falls back to the current Python-side rows.

        Returns:
            Displayed row dictionaries in the same order the user sees them.

        Raises:
            RuntimeError: If the browser returns a non-list result.
        """
        if self._grid is None:
            return self.get_rows()

        grid_id = int(self._grid.id)
        script = f"""
            (() => {{
                const grid = getElement({grid_id});
                if (!grid || !grid.api) {{
                    throw new Error('AG Grid API is not available for table widget {grid_id}');
                }}
                const rows = [];
                grid.api.forEachNodeAfterFilterAndSort(node => rows.push(node.data));
                return rows;
            }})()
            """
        rows = await ui.run_javascript(script, timeout=5.0)
        if not rows:
            return []
        if not isinstance(rows, list):
            raise RuntimeError(f'Expected AG Grid displayed rows as list, got {type(rows).__name__}')
        return [dict(row) for row in rows if isinstance(row, dict)]

    async def get_displayed_table_as_text(self) -> str:
        """Return browser-displayed rows as tab-separated text.

        The exported rows reflect AG Grid's browser-side filtering and sorting.
        Column order and visibility use the widget's current column definitions.

        Returns:
            TSV-formatted text for the displayed row set.
        """
        return self._rows_to_tsv(await self.get_displayed_rows())

    async def copy_table_data_to_clipboard(self) -> None:
        """Copy browser-displayed table rows to the clipboard as TSV.

        Rows are read from AG Grid after filtering and sorting. If the grid has
        not been built yet, the internal Python-side rows are copied instead.

        Returns:
            None.
        """
        text = await self.get_displayed_table_as_text()
        try:
            copy_to_clipboard(text)
        except RuntimeError as exc:
            logger.warning('unable to copy table data: %s', exc)
            ui.notify(str(exc), type='negative')
            return
        ui.notify('Table data copied to clipboard', type='positive')

    def _rows_to_tsv(self, rows: Sequence[Mapping[str, Any]]) -> str:
        """Format table rows as TSV using current visible columns.

        Args:
            rows: Row dictionaries to export.

        Returns:
            TSV-formatted text containing one header row and one row per input
            row. Returns an empty string when ``rows`` is empty.
        """
        if not rows:
            return ''

        visible_columns = [c for c in self._column_defs if not bool(c.get('hide', False))]
        headers = [str(c.get('headerName', c.get('field', ''))) for c in visible_columns]
        fields = [str(c.get('field', '')) for c in visible_columns]

        lines = ['\t'.join(self._sanitize_table_text_cell(header) for header in headers)]
        for row in rows:
            values = [self._sanitize_table_text_cell(row.get(field, '')) for field in fields]
            lines.append('\t'.join(values))
        return '\n'.join(lines)

    @staticmethod
    def _sanitize_table_text_cell(value: Any) -> str:
        """Return a TSV-safe string for one table cell.

        Args:
            value: Cell value to stringify.

        Returns:
            String with tabs and line breaks normalized to spaces.
        """
        if value is None:
            return ''
        return str(value).replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

    def _find_column_def(self, field: str) -> dict[str, Any]:
        for c in self._column_defs:
            if c.get('field') == field:
                return c
        raise ValueError(f'Unknown column field {field!r}')

    def _row_selection_object(self, selection: SelectionMode) -> dict[str, Any] | None:
        if selection == 'none':
            return None
        if selection == 'single':
            return {'mode': 'singleRow', 'enableClickSelection': True, 'checkboxes': False}
        return {'mode': 'multiRow', 'enableClickSelection': True, 'checkboxes': False}

    def _build_aggrid_options(self) -> dict[str, Any]:
        default_col_def: dict[str, Any] = {'sortable': True, 'filter': True, 'resizable': True}
        px = self._config.cell_font_size_px
        if px is not None:
            try:
                n = int(px)
            except (TypeError, ValueError):
                n = None
            else:
                if n >= 1:
                    fs = f'{n}px'
                    default_col_def['cellStyle'] = {'fontSize': fs}
                    default_col_def['headerStyle'] = {'fontSize': fs}

        base: dict[str, Any] = {
            'columnDefs': copy.deepcopy(self._column_defs),
            'rowData': [dict(r) for r in self._rows],
            'defaultColDef': default_col_def,
            ':getRowId': _get_row_id_js_expression(self._row_id_field),
            ':onRowClicked': js_on_row_clicked(emit_event=self._evt_select, row_id_field=self._row_id_field),
            # AG Grid Enterprise ships its own right-click menu (Copy, Copy with
            # Headers, Export, etc.) and intercepts the contextmenu event before
            # NiceGUI's ui.context_menu can see it. TableWidget owns the
            # right-click menu via _build_context_menu_content, so suppress AG
            # Grid's menu and ask AG Grid to also block the browser's native menu
            # over the grid surface.
            'suppressContextMenu': True,
            'preventDefaultOnContextMenu': True,
        }
        rh = self._config.row_height
        if rh is not None:
            try:
                rh_i = int(rh)
            except (TypeError, ValueError):
                rh_i = 0
            if rh_i >= 1:
                base['rowHeight'] = rh_i
        hh = self._config.header_height
        if hh is not None:
            try:
                hh_i = int(hh)
            except (TypeError, ValueError):
                hh_i = 0
            if hh_i >= 1:
                base['headerHeight'] = hh_i
        if self._config.stop_editing_when_cells_lose_focus:
            base['stopEditingWhenCellsLoseFocus'] = True
        if self._config.fit_columns_on_grid_resize:
            base[':onGridSizeChanged'] = 'params => params.api.sizeColumnsToFit()'
        row_sel = self._row_selection_object(self._config.selection_mode)
        if row_sel is not None:
            base['rowSelection'] = row_sel
        if self._config.enable_keyboard_row_nav and self._config.selection_mode != 'none':
            base[':onCellKeyDown'] = js_on_cell_key_down_select_prev_next(
                emit_event=self._evt_select,
                row_id_field=self._row_id_field,
            )
        if self._config.enable_edit_on_double_click:
            base[':onCellDoubleClicked'] = js_on_cell_double_clicked_start_editing()
            base[':onCellEditingStopped'] = js_on_cell_editing_stopped_emit_change(
                emit_event=self._evt_edit,
                row_id_field=self._row_id_field,
            )
        merged = _deep_merge_aggrid_options(base, self._config.extra_grid_options)
        return _deep_merge_aggrid_options(merged, self._grid_options_user)

    def _push_row_data_to_grid(self) -> None:
        if self._grid is None:
            return
        opts = dict(self._grid.options)
        opts['rowData'] = [dict(r) for r in self._rows]
        self._grid.options = opts
        self._grid.update()

    def _apply_column_defs_to_grid(self) -> None:
        if self._grid is None:
            return
        opts = dict(self._grid.options)
        opts['columnDefs'] = copy.deepcopy(self._column_defs)
        self._grid.options = opts
        self._grid.update()

    def _build_context_menu_content(self) -> None:
        """Populate the right-click context menu.

        Caller-provided actions are shown first, followed by the generic copy
        action, then the existing column-visibility toggles.

        Returns:
            None.
        """
        if self._on_build_context_menu is not None:
            self._on_build_context_menu(self)
            ui.separator()

        ui.menu_item('Copy Table Data', on_click=self.copy_table_data_to_clipboard)
        ui.separator()

        check = '✓'
        for c in self._column_defs:
            field = str(c['field'])
            header = str(c.get('headerName', field))
            visible = not bool(c.get('hide', False))
            label = f'{check} {header}' if visible else f'  {header}'
            ui.menu_item(label, on_click=lambda f=field: self.toggle_column_visible(f))

    def _on_context_menu_event(self, _e: events.GenericEventArguments) -> None:
        if self._context_menu is None:
            return
        with self._context_menu.clear():
            self._build_context_menu_content()

    def _on_select_emitted(self, e: events.GenericEventArguments) -> None:
        if self._config.selection_mode == 'none':
            return
        if getattr(self, '_selection_origin', 'internal') != 'internal':
            return
        args: dict[str, Any] = e.args or {}
        row_id = args.get('rowId')
        row_data = args.get('data') or {}
        if row_id is None:
            return
        row_id_str = str(row_id)
        if self._config.selection_mode == 'single':
            if row_id_str == self._last_selected_row_id:
                return
            self._selected_row_ids = [row_id_str]
            self._selected_rows = [dict(row_data)] if isinstance(row_data, dict) else []
            self._last_selected_row_id = row_id_str
        else:
            if row_id_str not in self._selected_row_ids:
                self._selected_row_ids.append(row_id_str)
                if isinstance(row_data, dict):
                    self._selected_rows.append(dict(row_data))
            self._last_selected_row_id = row_id_str

        if self._on_row_selected is not None and isinstance(row_data, dict) and row_data:
            self._on_row_selected(dict(row_data))

    def _on_edit_emitted(self, e: events.GenericEventArguments) -> None:
        args: dict[str, Any] = e.args or {}
        row_id = args.get('rowId')
        col_id = args.get('colId')
        old_value = args.get('oldValue')
        new_value = args.get('newValue')
        data = args.get('data') or {}
        if row_id is None or col_id is None:
            return
        row_id_str = str(row_id)
        field = str(col_id)
        if old_value == new_value:
            return
        for i, row in enumerate(self._rows):
            if str(row.get(self._row_id_field)) == row_id_str:
                self._rows[i][field] = new_value
                break
        if self._on_cell_edited is not None and isinstance(data, dict):
            self._on_cell_edited(row_id_str, field, old_value, new_value, dict(data))
