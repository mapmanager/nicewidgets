"""Composable NiceGUI ``ui.aggrid`` tree widget with id-keyed updates.

The widget mirrors the *used* subset of ``TableWidget`` public API so external
NiceGUI consumers can migrate with minimal glue changes, while adding tree-specific
methods for subtree replacement and expand/collapse controls.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from nicegui import events, ui

from nicewidgets.aggrid_common.column_def import ColumnDef
from nicewidgets.tree_widget.config import TreeWidgetConfig, font_scaled_column_width_px
from nicewidgets.tree_widget.js_hooks import (
    js_on_cell_key_down_select_prev_next,
    js_on_row_clicked,
    js_on_row_group_closed,
    js_on_row_group_opened,
)
from nicewidgets.utils.desktop import is_pywebview_desktop
from nicewidgets.utils.logging import get_logger

try:
    import pyperclip
except ImportError:  # pragma: no cover - native-only optional dependency
    pyperclip = None  # type: ignore[assignment]

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


def _index_column_value_getter_js() -> str:
    """Return AG Grid ``valueGetter`` for 1-based root-only row indices.

    Uses ``node.level === 0`` so child rows stay blank, and ``forEachNode`` so
    indices follow row-model (load) order rather than visible render order.
    """
    return (
        '(params) => {'
        'const node = params.node;'
        'if (!node || node.level !== 0) return "";'
        'const roots = [];'
        'params.api.forEachNode(n => { if (n.level === 0) roots.push(n); });'
        'const i = roots.indexOf(node);'
        'return i === -1 ? "" : i + 1;'
        '}'
    )


def _auto_inject_show_row_group(column_defs: list[dict[str, Any]]) -> None:
    """Set ``showRowGroup: True`` on the column hosting the group cell renderer.

    AG Grid's ``groupDisplayType: 'custom'`` mode (used for tree data when
    no ``autoGroupColumnDef`` is provided) only renders disclosure chevrons
    inside a column that has BOTH ``cellRenderer: 'agGroupCellRenderer'``
    AND ``showRowGroup: True``. Callers typically remember the cellRenderer
    but forget ``showRowGroup``; this helper sets it automatically on the
    first column whose cellRenderer is ``'agGroupCellRenderer'``.

    The function mutates ``column_defs`` in place. Columns that already have
    a truthy ``showRowGroup`` are left untouched.

    Args:
        column_defs: AG Grid column definition dicts (mutated in place).
    """
    for col in column_defs:
        if col.get('cellRenderer') != 'agGroupCellRenderer':
            continue
        if col.get('showRowGroup'):
            return
        col['showRowGroup'] = True
        logger.debug('auto-injected showRowGroup on column %r', col.get('field'))
        return


class TreeWidget:
    """AG Grid Enterprise tree wrapper with a consumer-friendly public API."""

    def __init__(
        self,
        columns: Sequence[ColumnDef],
        row_id_field: str,
        rows: Sequence[Mapping[str, Any]] | None = None,
        *,
        on_row_selected: Callable[[dict[str, Any]], None] | None = None,
        on_build_context_menu: Callable[[TreeWidget], None] | None = None,
        config: TreeWidgetConfig | None = None,
        grid_options: Mapping[str, Any] | None = None,
        path_field: str = 'hierarchy_path',
        auto_group_column_def: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize tree widget state without creating UI elements.

        Args:
            columns: Column definitions.
            row_id_field: Row key containing stable unique row id.
            rows: Initial row set.
            on_row_selected: Callback fired when a row is selected/clicked.
            on_build_context_menu: Callback for caller menu items.
            config: Tree widget behavior/configuration.
            grid_options: Additional AG Grid options merged last.
            path_field: Row key containing tree path list.
            auto_group_column_def: Optional AG Grid ``autoGroupColumnDef``.
        """
        validate_row_id_field(row_id_field)
        if not path_field or not str(path_field).strip():
            raise ValueError('path_field must be a non-empty string')

        self._row_id_field = row_id_field
        self._path_field = path_field
        self._on_row_selected = on_row_selected
        self._on_build_context_menu = on_build_context_menu
        self._config = config or TreeWidgetConfig()
        self._grid_options_user = dict(grid_options or {})
        self._auto_group_column_def = dict(auto_group_column_def) if auto_group_column_def is not None else None
        self._selection_origin = 'internal'

        self._evt_select = f'tree_widget_select_{id(self)}'
        self._evt_expand = f'tree_widget_expand_{id(self)}'

        self._index_field: str | None = None
        if self._config.show_index_column:
            idx_f = str(self._config.index_field).strip()
            if not idx_f:
                raise ValueError('index_field must be non-empty when show_index_column is true')
            for c in columns:
                if c.field == idx_f:
                    raise ValueError(
                        f'Column field {idx_f!r} conflicts with TreeWidgetConfig.index_field; '
                        'rename the column or set a different index_field'
                    )
            self._index_field = idx_f
            idx_width = font_scaled_column_width_px(self._config.cell_font_size_px)
            scale = float(self._config.index_column_width_multiplier)
            if scale != 1.0:
                idx_width = max(1, int(round(idx_width * scale)))
            index_col = ColumnDef(
                field=idx_f,
                headerName=str(self._config.index_header),
                extra={
                    'editable': False,
                    'sortable': False,
                    'filter': False,
                    'resizable': True,
                    'width': idx_width,
                    'minWidth': idx_width,
                    ':valueGetter': _index_column_value_getter_js(),
                },
            )
            built_columns = (index_col, *columns)
        else:
            built_columns = tuple(columns)

        self._column_defs: list[dict[str, Any]] = [c.as_aggrid_column_def() for c in built_columns]
        self._rows: list[dict[str, Any]] = [dict(r) for r in (rows or ())]
        validate_rows_for_row_id_field(self._rows, self._row_id_field)

        if self._auto_group_column_def is None:
            _auto_inject_show_row_group(self._column_defs)

        self._known_ids_by_group: dict[str, set[str]] = {}
        for row in self._rows:
            self._track_added(row)

        self._selected_row_ids: list[str] = []
        self._selected_rows: list[dict[str, Any]] = []
        self._last_selected_row_id: str | None = None
        self._expanded_group_ids: set[str] = set()

        self._root: ui.column | None = None
        self._grid: ui.aggrid | None = None
        self._context_menu: ui.context_menu | None = None

        ui.on(self._evt_select, self._on_select_emitted)
        ui.on(self._evt_expand, self._on_expand_emitted)

    def build(self, parent: ui.element | None = None) -> ui.column:
        """Create the wrapper + context menu + AG Grid under ``parent``.

        Args:
            parent: Optional parent element; when omitted a default sized
                container is created.

        Returns:
            Root column containing the tree grid.
        """
        if self._config.enterprise_module_url:
            ui.aggrid.set_module_source(self._config.enterprise_module_url)

        container = parent if parent is not None else ui.column().classes('w-full').style('height: 24rem;')
        with container:
            self._root = ui.column().classes('w-full h-full min-w-0 min-h-0')
            with self._root:
                self._context_menu = ui.context_menu()
                with self._context_menu:
                    self._build_context_menu_content()
                self._root.on('contextmenu', self._on_context_menu_event)
                # The AG Grid element is created lazily, only once rows exist.
                # A grid created with empty ``rowData`` and later filled accepts
                # programmatic ``setSelected`` state but never repaints the
                # selected row (verified in-browser). Creating it born with rows
                # avoids that broken state entirely.
                self._ensure_grid_built()
        return self._root

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable pointer interaction with the tree widget.

        Disabling toggles ``pointer-events-none opacity-60`` on the root
        container only; the inner AG Grid element is intentionally not
        re-pushed. Re-pushing element state to the client via
        ``self._grid.update()`` causes AG Grid to re-render and drop
        client-side state (notably tree-group expansion), which is
        surprising during transient busy-state cycles. The CSS overlay
        on the root already blocks pointer input across the entire
        widget surface.

        Args:
            enabled: Desired enabled state.
        """
        enabled = bool(enabled)
        if self._root is None:
            return
        self._root.enabled = enabled
        if enabled:
            self._root.classes(remove='pointer-events-none opacity-60')
        else:
            self._root.classes(add='pointer-events-none opacity-60')
        self._root.update()

    def get_selected_rows(self) -> list[dict[str, Any]]:
        """Return last known selected rows."""
        return [dict(r) for r in self._selected_rows]

    def set_selected_row_ids(self, row_ids: Sequence[str], *, origin: str = 'external') -> None:
        """Programmatically select rows by row id (selection mode aware).

        Args:
            row_ids: Desired row ids to select.
            origin: Selection origin marker to avoid event echo loops.
        """
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

        # Idempotent guard: when the requested selection already matches the
        # tracked selection, the grid already reflects it (from a native click
        # or a prior programmatic selection, both of which survive id-keyed
        # ``applyTransaction`` updates). Re-issuing ``deselectAll`` +
        # ``setSelected`` in that case is pure churn and produces a visible
        # deselect/reselect flash when repeated selection syncs fire per click.
        already_selected = keep == self._selected_row_ids

        self._selected_row_ids = keep
        self._selected_rows = [row_by_id[rid] for rid in keep]
        self._last_selected_row_id = keep[0] if keep else None

        if self._grid is None:
            return
        if already_selected:
            return

        self._selection_origin = origin
        if not keep:
            self._grid.run_grid_method('deselectAll')
        elif self._config.selection_mode == 'single':
            self._grid.run_row_method(keep[0], 'setSelected', True, True)
        else:
            self._grid.run_grid_method('deselectAll')
            for rid in keep:
                self._grid.run_row_method(rid, 'setSelected', True, False)
        self._selection_origin = 'internal'

    def clear_selection(self) -> None:
        """Clear selected-row tracking and grid selection."""
        self._selected_row_ids = []
        self._selected_rows = []
        self._last_selected_row_id = None
        if self._grid is not None:
            self._grid.run_grid_method('deselectAll')

    def scroll_row_id_into_view(self, row_id: str) -> None:
        """Expand and scroll the tree so the requested row is visible.

        This is intended for programmatic selection driven from outside the
        tree (for example a pool-plot click). It is intentionally NOT called
        by :meth:`set_selected_row_ids`, so a user clicking a row in the tree
        never triggers an automatic scroll.

        The method resolves the row by its stable AG Grid row id, uses AG Grid's
        public ``setRowNodeExpanded`` API to expand the row and all ancestors
        synchronously, and then scrolls the actual target row to the middle of
        the viewport. JavaScript is sent through the grid element's owning
        client so the method is safe when invoked from an async background-task
        completion without an active NiceGUI slot context. Unknown row ids and
        unbuilt grids are no-ops.

        Args:
            row_id: Stable row id to reveal.
        """
        if self._grid is None:
            return
        rid = str(row_id)
        if not rid:
            return
        grid_id = int(self._grid.id)
        rid_literal = json.dumps(rid)
        script = f"""
            (() => {{
                const grid = getElement({grid_id});
                if (!grid || !grid.api) return;
                const target = grid.api.getRowNode({rid_literal});
                if (!target) return;

                grid.api.setRowNodeExpanded(
                    target,
                    true,
                    true,
                    {{forceSync: true}},
                );
                grid.api.ensureNodeVisible(target, 'middle');
            }})()
        """
        self._grid.client.run_javascript(script)

    def set_data(self, rows: Sequence[Mapping[str, Any]]) -> None:
        """Replace all rows and refresh the tree.

        Args:
            rows: New complete row set.
        """
        new_rows = [dict(r) for r in rows]
        validate_rows_for_row_id_field(new_rows, self._row_id_field)
        self._rows = new_rows
        self._known_ids_by_group.clear()
        for row in self._rows:
            self._track_added(row)
        if self._grid is None:
            self._ensure_grid_built()
        else:
            self._push_row_data_to_grid()
        if self._config.clear_selection_on_set_data:
            self.clear_selection()

    def update_row(self, row_id: str, row: Mapping[str, Any]) -> None:
        """Update a single row by id, patching the row node when possible.

        Args:
            row_id: Existing row id to replace.
            row: Replacement row.

        Raises:
            ValueError: If no row exists with ``row_id``.
        """
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

        old = self._rows[idx]
        self._rows[idx] = replacement
        self._track_removed(old)
        self._track_added(replacement)

        if self._grid is None:
            self._ensure_grid_built()
            return
        try:
            self._grid.run_row_method(rid, 'setData', dict(self._rows[idx]))
        except RuntimeError:
            self._push_row_data_to_grid()

    def replace_group_rows(self, group_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Replace every row in one top-level group via AG Grid transaction.

        Only rows whose data changed are included in the update transaction.
        Identical replacement data updates Python-side ordering without sending
        browser grid commands.

        Args:
            group_id: Top-level group id (value of ``path_field[0]``).
            rows: Complete replacement row set for that group.
        """
        rows_list = [dict(r) for r in rows]
        validate_rows_for_row_id_field(rows_list, self._row_id_field)

        old_group_rows = [
            dict(row)
            for row in self._rows
            if str(row.get(self._row_id_field))
            in self._known_ids_by_group.get(group_id, set())
        ]
        old_rows_by_id = {
            str(row[self._row_id_field]): row for row in old_group_rows
        }
        new_rows_by_id = {
            str(row[self._row_id_field]): row for row in rows_list
        }
        old_ids = set(old_rows_by_id)
        new_ids = set(new_rows_by_id)

        rows_to_add = [new_rows_by_id[rid] for rid in new_ids - old_ids]
        rows_to_update = [
            new_rows_by_id[rid]
            for rid in new_ids & old_ids
            if old_rows_by_id[rid] != new_rows_by_id[rid]
        ]
        ids_to_remove = old_ids - new_ids

        new_all_rows: list[dict[str, Any]] = []
        replaced = False
        for row in self._rows:
            rid = str(row.get(self._row_id_field))
            if rid in old_ids:
                if not replaced:
                    new_all_rows.extend(rows_list)
                    replaced = True
                continue
            new_all_rows.append(row)
        if not replaced:
            new_all_rows.extend(rows_list)
        self._rows = new_all_rows
        if new_ids:
            self._known_ids_by_group[group_id] = new_ids
        else:
            self._known_ids_by_group.pop(group_id, None)

        if ids_to_remove:
            self._selected_row_ids = [
                rid for rid in self._selected_row_ids if rid not in ids_to_remove
            ]
            row_by_id = {str(row[self._row_id_field]): row for row in self._rows}
            self._selected_rows = [
                dict(row_by_id[rid])
                for rid in self._selected_row_ids
                if rid in row_by_id
            ]
            self._last_selected_row_id = (
                self._selected_row_ids[0] if self._selected_row_ids else None
            )

        if self._grid is None:
            self._ensure_grid_built()
            return

        transaction: dict[str, Any] = {}
        if rows_to_add:
            transaction['add'] = rows_to_add
        if rows_to_update:
            transaction['update'] = rows_to_update
        if ids_to_remove:
            transaction['remove'] = [
                {self._row_id_field: rid} for rid in ids_to_remove
            ]
        if not transaction:
            return

        self._grid.run_grid_method('applyTransaction', transaction)
        if rows_to_add:
            self.expand_group(group_id)

    def expand_all_nodes(self) -> None:
        """Expand every tree group on the client."""
        if self._grid is None:
            return
        self._grid.run_grid_method('expandAll')
        for row in self._rows:
            row_id = row.get(self._row_id_field)
            if isinstance(row_id, str) and row_id:
                self._expanded_group_ids.add(row_id)

    def collapse_all_nodes(self) -> None:
        """Collapse every tree group on the client."""
        self._expanded_group_ids.clear()
        if self._grid is None:
            return
        self._grid.run_grid_method('collapseAll')

    def expand_group(self, group_id: str) -> None:
        """Expand one tree group by row id.

        Args:
            group_id: Row id of the depth-1 group row.
        """
        self._expanded_group_ids.add(group_id)
        if self._grid is None:
            return
        self._grid.run_row_method(group_id, 'setExpanded', True)

    def expanded_group_ids(self) -> frozenset[str]:
        """Return file-group row ids currently expanded in the tree.

        Returns:
            Frozen set of expanded group row ids tracked in Python.
        """
        return frozenset(self._expanded_group_ids)

    async def get_displayed_rows(self) -> list[dict[str, Any]]:
        """Return AG Grid rows after browser-side filtering/sorting.

        Returns:
            Displayed row dictionaries in user-visible order.

        Raises:
            RuntimeError: If browser JS returns a non-list.
        """
        if self._grid is None:
            return [dict(r) for r in self._rows]

        grid_id = int(self._grid.id)
        script = f"""
            (() => {{
                const grid = getElement({grid_id});
                if (!grid || !grid.api) {{
                    throw new Error('AG Grid API is not available for tree widget {grid_id}');
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
            'treeData': True,
            'columnDefs': copy.deepcopy(self._column_defs),
            'rowData': [dict(r) for r in self._rows],
            'defaultColDef': default_col_def,
            ':getDataPath': f'data => data.{self._path_field}',
            ':getRowId': _get_row_id_js_expression(self._row_id_field),
            ':onRowClicked': js_on_row_clicked(emit_event=self._evt_select, row_id_field=self._row_id_field),
            # AG Grid Enterprise ships its own right-click menu (Copy, Copy
            # with Headers, Export, etc.) and intercepts the contextmenu
            # event before NiceGUI's ui.context_menu can see it. The
            # TreeWidget owns the right-click menu via on_build_context_menu,
            # so suppress AG Grid's menu globally and ask AG Grid to also
            # block the browser's native menu over the grid surface.
            'suppressContextMenu': True,
            'preventDefaultOnContextMenu': True,
            'suppressRowHoverHighlight': True,
        }
        if self._auto_group_column_def is not None:
            base['autoGroupColumnDef'] = dict(self._auto_group_column_def)
        else:
            # AG Grid's `groupDisplayType` (type ``RowGroupingDisplayType``,
            # values: 'singleColumn' | 'multipleColumns' | 'groupRows' |
            # 'custom') is documented to apply to BOTH row grouping AND tree
            # data. Setting it to 'custom' tells AG Grid not to auto-create a
            # group column; it then renders the disclosure chevron in the
            # caller's column that has `showRowGroup: True` plus
            # `cellRenderer: 'agGroupCellRenderer'`. The widget injects
            # `showRowGroup: True` automatically in __init__ on the column
            # whose cellRenderer is `agGroupCellRenderer` so callers do not
            # have to remember to set it.
            base['groupDisplayType'] = 'custom'

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
        if self._config.fit_columns_on_grid_resize:
            base[':onGridSizeChanged'] = 'params => params.api.sizeColumnsToFit()'
        if self._config.suppress_movable_columns:
            base['suppressMovableColumns'] = True
            default_col_def['suppressMovable'] = True
        if self._config.selection_mode != 'none':
            mode = 'singleRow' if self._config.selection_mode == 'single' else 'multiRow'
            base['rowSelection'] = {'mode': mode, 'enableClickSelection': True, 'checkboxes': False}
        if self._config.enable_keyboard_row_nav and self._config.selection_mode != 'none':
            base[':onCellKeyDown'] = js_on_cell_key_down_select_prev_next(
                emit_event=self._evt_select,
                row_id_field=self._row_id_field,
            )
        base[':onRowGroupOpened'] = js_on_row_group_opened(
            emit_event=self._evt_expand,
            row_id_field=self._row_id_field,
        )
        base[':onRowGroupClosed'] = js_on_row_group_closed(
            emit_event=self._evt_expand,
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

    def _ensure_grid_built(self) -> None:
        """Create the AG Grid element born with current rows, if needed.

        No-op when the root container does not exist yet, when the grid is
        already built, or when there are no rows to show. Creating the grid only
        once rows exist guarantees it is never born with empty ``rowData`` (a
        state in which programmatic selection updates AG Grid selection state but
        never repaints the selected row). The context menu and row-select event
        wiring are unaffected: the menu lives on ``self._root`` and selection
        events are delivered via the module-level ``ui.on`` handler, not the
        grid element instance.
        """
        if self._root is None or self._grid is not None or not self._rows:
            return
        with self._root:
            self._grid = ui.aggrid(
                self._build_aggrid_options(),
                auto_size_columns=self._config.auto_size_columns,
                modules='enterprise',
            ).classes('w-full h-full min-w-0 min-h-0').style('height: 100%;')

    def _find_column_def(self, field: str) -> dict[str, Any]:
        for c in self._column_defs:
            if c.get('field') == field:
                return c
        raise ValueError(f'Unknown column field {field!r}')

    def _set_column_visible(self, field: str, visible: bool) -> None:
        col = self._find_column_def(field)
        col['hide'] = not visible
        if self._grid is None:
            return
        opts = dict(self._grid.options)
        opts['columnDefs'] = copy.deepcopy(self._column_defs)
        self._grid.options = opts
        self._grid.update()

    def _build_context_menu_content(self) -> None:
        if self._on_build_context_menu is not None:
            self._on_build_context_menu(self)
            ui.separator()

        ui.menu_item('Expand All', on_click=self.expand_all_nodes)
        ui.menu_item('Collapse All', on_click=self.collapse_all_nodes)
        ui.separator()

        ui.menu_item('Copy Table Data', on_click=self._copy_table_data_to_clipboard)
        ui.separator()

        check = '✓'
        for c in self._column_defs:
            field = str(c['field'])
            if self._index_field is not None and field == self._index_field:
                header = str(self._config.index_menu_label)
            else:
                header = str(c.get('headerName') or field)
            visible = not bool(c.get('hide', False))
            label = f'{check} {header}' if visible else f'  {header}'
            ui.menu_item(label, on_click=lambda f=field, v=visible: self._set_column_visible(f, not v))

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

    def _on_expand_emitted(self, e: events.GenericEventArguments) -> None:
        """Track expanded/collapsed tree groups from AG Grid callbacks.

        Args:
            e: NiceGUI event payload from the expand JS hook.

        Returns:
            None.
        """
        args: dict[str, Any] = e.args or {}
        row_id = args.get('rowId')
        if not isinstance(row_id, str) or not row_id:
            return
        if bool(args.get('expanded')):
            self._expanded_group_ids.add(row_id)
        else:
            self._expanded_group_ids.discard(row_id)

    def _track_added(self, row: Mapping[str, Any]) -> None:
        path = row.get(self._path_field) or []
        if not path:
            return
        group_id = str(path[0])
        self._known_ids_by_group.setdefault(group_id, set()).add(str(row[self._row_id_field]))

    def _track_removed(self, row: Mapping[str, Any]) -> None:
        path = row.get(self._path_field) or []
        if not path:
            return
        group_id = str(path[0])
        rid = str(row[self._row_id_field])
        bucket = self._known_ids_by_group.get(group_id)
        if bucket is None:
            return
        bucket.discard(rid)
        if not bucket:
            del self._known_ids_by_group[group_id]

    async def _copy_table_data_to_clipboard(self) -> None:
        """Copy displayed rows to clipboard from a context-menu click.

        Two paths:

        * Native window (pywebview) -- ``navigator.clipboard`` is unavailable
          and there is no user-gesture concept; copy via ``pyperclip`` against
          the current Python-side rows.
        * Browser -- ``navigator.clipboard.writeText`` requires the call to
          stay inside the originating user-gesture context, so we fold both
          the displayed-rows fetch and the clipboard write into a single
          ``ui.run_javascript`` round-trip.
        """
        if is_pywebview_desktop():
            text = self._rows_to_tsv(self._rows)
            if pyperclip is None:
                logger.warning('pyperclip is required for native clipboard support')
                ui.notify('pyperclip is required for native clipboard support', type='negative')
                return
            pyperclip.copy(text)
            ui.notify('Tree data copied to clipboard', type='positive')
            return

        if self._grid is None:
            text = self._rows_to_tsv(self._rows)
            ui.run_javascript(f'navigator.clipboard.writeText({json.dumps(text)});')
            ui.notify('Tree data copied to clipboard', type='positive')
            return

        script = self._build_browser_copy_script()
        try:
            result = await ui.run_javascript(script, timeout=5.0)
        except Exception as exc:
            logger.warning('clipboard JS failed: %s', exc)
            ui.notify(f'Copy failed: {exc}', type='negative')
            return

        if isinstance(result, dict) and result.get('ok'):
            ui.notify('Tree data copied to clipboard', type='positive')
            return
        err = (result or {}).get('error') if isinstance(result, dict) else None
        ui.notify(f'Copy failed{f": {err}" if err else ""}', type='negative')

    def _build_browser_copy_script(self) -> str:
        """Build the single-roundtrip JS snippet copying displayed rows to clipboard.

        Returns:
            JavaScript source. Reads displayed rows from the grid (filter +
            sort applied), builds a TSV using the widget's visible columns,
            and writes it to the system clipboard. Falls back to the legacy
            ``document.execCommand('copy')`` API when ``navigator.clipboard``
            is unavailable (e.g. older browsers, non-secure contexts).
        """
        grid_id = int(self._grid.id) if self._grid is not None else 0
        visible_columns = [c for c in self._column_defs if not bool(c.get('hide', False))]
        headers = [str(c.get('headerName', c.get('field', ''))) for c in visible_columns]
        fields = [str(c.get('field', '')) for c in visible_columns]
        headers_lit = json.dumps(headers)
        fields_lit = json.dumps(fields)
        return f"""
            (async () => {{
                const grid = getElement({grid_id});
                if (!grid || !grid.api) {{
                    return {{ ok: false, error: 'AG Grid API unavailable' }};
                }}
                const rows = [];
                grid.api.forEachNodeAfterFilterAndSort(n => rows.push(n.data));
                const headers = {headers_lit};
                const fields = {fields_lit};
                const sanitize = v => (v == null ? '' : String(v).replace(/[\\t\\r\\n]/g, ' '));
                const headerLine = headers.map(sanitize).join('\\t');
                const bodyLines = rows.map(r => fields.map(f => sanitize(r ? r[f] : '')).join('\\t'));
                const tsv = rows.length ? headerLine + '\\n' + bodyLines.join('\\n') : headerLine;
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    try {{
                        await navigator.clipboard.writeText(tsv);
                        return {{ ok: true }};
                    }} catch (err) {{
                        // fall through to execCommand fallback below
                        var lastErr = err && err.message ? err.message : String(err);
                    }}
                }}
                try {{
                    const ta = document.createElement('textarea');
                    ta.value = tsv;
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.select();
                    const ok = document.execCommand('copy');
                    document.body.removeChild(ta);
                    if (ok) return {{ ok: true }};
                    const fallbackErr = (typeof lastErr !== 'undefined') ? lastErr : 'execCommand copy failed';
                    return {{ ok: false, error: fallbackErr }};
                }} catch (err) {{
                    return {{ ok: false, error: err && err.message ? err.message : String(err) }};
                }}
            }})()
        """.strip()

    def _rows_to_tsv(self, rows: Sequence[Mapping[str, Any]]) -> str:
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
        if value is None:
            return ''
        return str(value).replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
