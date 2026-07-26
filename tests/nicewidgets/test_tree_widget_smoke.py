"""Smoke tests for ``nicewidgets.tree_widget`` (no live NiceGUI client)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from nicewidgets.aggrid_common.column_def import ColumnDef
from nicewidgets.tree_widget.config import TreeWidgetConfig, font_scaled_column_width_px, scaled_row_header_heights_px
from nicewidgets.tree_widget.tree_widget import (
    TreeWidget,
    _auto_inject_show_row_group,
    _get_row_id_js_expression,
    _index_column_value_getter_js,
    validate_row_id_field,
    validate_rows_for_row_id_field,
)
from nicewidgets.tree_widget import tree_widget


def _sample_rows() -> list[dict[str, Any]]:
    return [
        {'row_id': '/a', 'hierarchy_path': ['/a'], 'name': 'A'},
        {'row_id': '/a::1', 'hierarchy_path': ['/a', '/a::1'], 'name': 'A1'},
        {'row_id': '/b', 'hierarchy_path': ['/b'], 'name': 'B'},
    ]


def _sample_columns() -> list[ColumnDef]:
    return [
        ColumnDef(field='row_id', headerName='Row Id', hide=True),
        ColumnDef(field='name', headerName='Name'),
    ]


def test_validate_row_id_field_rejects_empty() -> None:
    with pytest.raises(ValueError, match='row_id_field'):
        validate_row_id_field('')
    with pytest.raises(ValueError, match='row_id_field'):
        validate_row_id_field('   ')


def test_validate_rows_missing_key() -> None:
    with pytest.raises(ValueError, match='missing'):
        validate_rows_for_row_id_field([{'a': 1}], 'id')


def test_validate_rows_non_str_id() -> None:
    with pytest.raises(ValueError, match='str'):
        validate_rows_for_row_id_field([{'id': 1}], 'id')


def test_validate_rows_duplicate_ids() -> None:
    with pytest.raises(ValueError, match='Duplicate'):
        validate_rows_for_row_id_field([{'id': 'x'}, {'id': 'x'}], 'id')


def test_tree_widget_config_defaults() -> None:
    cfg = TreeWidgetConfig()
    assert cfg.selection_mode == 'single'
    assert cfg.clear_selection_on_set_data is True
    assert cfg.enable_keyboard_row_nav is True
    assert cfg.auto_size_columns is True
    assert cfg.suppress_movable_columns is False
    assert cfg.show_index_column is False
    assert cfg.index_field == 'file_row_index'
    assert cfg.index_header == ''


def test_normalize_tree_theme() -> None:
    from nicewidgets.tree_widget.tree_widget import normalize_tree_theme

    assert normalize_tree_theme('dark') == 'dark'
    assert normalize_tree_theme('LIGHT') == 'light'


def test_tree_widget_theme_api_stores_before_build() -> None:
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
    )
    tw.set_dark_mode(True)
    assert tw._theme == 'dark'
    tw.set_theme('light')
    assert tw._theme == 'light'


def test_build_aggrid_options_suppress_movable_columns() -> None:
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(suppress_movable_columns=True),
    )
    opts = tw._build_aggrid_options()
    assert opts['suppressMovableColumns'] is True
    assert opts['defaultColDef']['suppressMovable'] is True


def test_scaled_row_header_heights_px_clamped() -> None:
    r, h = scaled_row_header_heights_px(13)
    assert r == 38 and h == 37


def test_font_scaled_column_width_px_scales_with_font() -> None:
    assert font_scaled_column_width_px(11) == 66
    assert font_scaled_column_width_px(None) == font_scaled_column_width_px(13)
    assert font_scaled_column_width_px(13) == 78


def test_get_row_id_js_expression_escapes_field_name() -> None:
    js = _get_row_id_js_expression('path')
    assert 'params.data' in js
    assert '"path"' in js


def test_build_aggrid_options_tree_defaults() -> None:
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
    )
    opts = tw._build_aggrid_options()
    assert opts['treeData'] is True
    assert ':getDataPath' in opts
    assert ':getRowId' in opts
    # AG Grid v34: `groupDisplayType: 'custom'` is the documented option
    # for both row grouping and tree data; `treeDataDisplayType` does NOT
    # accept 'custom' as a value and would be silently ignored.
    assert opts['groupDisplayType'] == 'custom'
    assert 'treeDataDisplayType' not in opts
    assert 'autoGroupColumnDef' not in opts
    # AG Grid Enterprise default context menu (Copy / Copy with Headers /
    # Export / ...) must be suppressed so the NiceGUI ui.context_menu owned
    # by the widget is the only menu the user sees. Browser default menu
    # over the grid surface is also prevented.
    assert opts['suppressContextMenu'] is True
    assert opts['preventDefaultOnContextMenu'] is True


def test_auto_inject_show_row_group_sets_flag_on_aggroupcellrenderer_column() -> None:
    """The column with cellRenderer agGroupCellRenderer gets showRowGroup=True."""
    cols = [
        {'field': 'a', 'headerName': 'A'},
        {'field': 'name', 'headerName': 'Name', 'cellRenderer': 'agGroupCellRenderer'},
        {'field': 'b', 'headerName': 'B'},
    ]
    _auto_inject_show_row_group(cols)
    assert cols[0].get('showRowGroup') is None
    assert cols[1]['showRowGroup'] is True
    assert cols[2].get('showRowGroup') is None


def test_auto_inject_show_row_group_is_idempotent() -> None:
    """If caller already set showRowGroup, do not overwrite."""
    cols = [
        {
            'field': 'name',
            'headerName': 'Name',
            'cellRenderer': 'agGroupCellRenderer',
            'showRowGroup': 'name',
        },
    ]
    _auto_inject_show_row_group(cols)
    assert cols[0]['showRowGroup'] == 'name'


def test_auto_inject_show_row_group_no_op_when_no_aggroupcellrenderer() -> None:
    """Callers without an agGroupCellRenderer column are untouched."""
    cols = [{'field': 'a', 'headerName': 'A'}]
    _auto_inject_show_row_group(cols)
    assert 'showRowGroup' not in cols[0]


def test_treewidget_init_auto_injects_show_row_group_on_group_renderer_column() -> None:
    """End-to-end: TreeWidget __init__ injects showRowGroup when needed."""
    cols = [
        ColumnDef(field='row_id', headerName='Row Id', hide=True),
        ColumnDef(
            field='name',
            headerName='Name',
            extra={'cellRenderer': 'agGroupCellRenderer'},
        ),
    ]
    tw = TreeWidget(columns=cols, row_id_field='row_id', rows=_sample_rows())
    name_col = next(c for c in tw._column_defs if c['field'] == 'name')
    assert name_col['showRowGroup'] is True


def test_treewidget_skips_show_row_group_inject_when_auto_group_column_def_set() -> None:
    """Caller-supplied autoGroupColumnDef path does not need showRowGroup injection."""
    cols = [
        ColumnDef(field='row_id', headerName='Row Id', hide=True),
        ColumnDef(
            field='name',
            headerName='Name',
            extra={'cellRenderer': 'agGroupCellRenderer'},
        ),
    ]
    tw = TreeWidget(
        columns=cols,
        row_id_field='row_id',
        rows=_sample_rows(),
        auto_group_column_def={'headerName': 'Group'},
    )
    name_col = next(c for c in tw._column_defs if c['field'] == 'name')
    assert 'showRowGroup' not in name_col


def test_build_browser_copy_script_includes_displayed_rows_and_clipboard() -> None:
    """Browser-mode copy must do row read + clipboard write in one JS call."""
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw._grid = SimpleNamespace(id=99)

    script = tw._build_browser_copy_script()

    assert 'getElement(99)' in script
    assert 'forEachNodeAfterFilterAndSort' in script
    assert 'navigator.clipboard.writeText' in script
    # Legacy fallback for non-secure contexts / older browsers.
    assert "execCommand('copy')" in script
    # Headers reflect visible columns only (row_id is hidden in sample).
    assert '"Name"' in script
    assert '"Row Id"' not in script


def test_copy_table_data_uses_pyperclip_in_native_window(monkeypatch: Any) -> None:
    """Native window path should bypass JS and use pyperclip on python rows."""
    import asyncio

    copied: list[str] = []
    monkeypatch.setattr(tree_widget, 'is_pywebview_desktop', lambda: True)
    monkeypatch.setattr(tree_widget, 'pyperclip', SimpleNamespace(copy=copied.append))
    monkeypatch.setattr(tree_widget.ui, 'notify', lambda *_a, **_k: None)

    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())

    asyncio.run(tw._copy_table_data_to_clipboard())

    assert copied  # exact TSV shape covered by browser-script test
    assert 'Name' in copied[0]


def test_copy_table_data_runs_browser_script_and_notifies_on_success(monkeypatch: Any) -> None:
    """Browser path should run the single-roundtrip JS and notify on ok=True."""
    import asyncio

    scripts: list[str] = []
    notifications: list[tuple[str, str]] = []

    async def fake_run_javascript(script: str, timeout: float = 1.0) -> dict[str, Any]:
        scripts.append(script)
        assert timeout == 5.0
        return {'ok': True}

    monkeypatch.setattr(tree_widget, 'is_pywebview_desktop', lambda: False)
    monkeypatch.setattr(tree_widget.ui, 'run_javascript', fake_run_javascript)
    monkeypatch.setattr(
        tree_widget.ui,
        'notify',
        lambda message, type='info': notifications.append((message, type)),
    )

    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw._grid = SimpleNamespace(id=99)

    asyncio.run(tw._copy_table_data_to_clipboard())

    assert len(scripts) == 1
    assert 'navigator.clipboard.writeText' in scripts[0]
    assert notifications == [('Tree data copied to clipboard', 'positive')]


def test_set_data_clears_selection_by_default() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw.set_selected_row_ids(['/a'])
    assert tw.get_selected_rows()[0]['row_id'] == '/a'
    tw.set_data([{'row_id': '/z', 'hierarchy_path': ['/z'], 'name': 'Z'}])
    assert tw.get_selected_rows() == []


def test_set_data_keeps_selection_when_configured() -> None:
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(clear_selection_on_set_data=False),
    )
    tw.set_selected_row_ids(['/a'])
    tw.set_data([{'row_id': '/z', 'hierarchy_path': ['/z'], 'name': 'Z'}])
    assert tw.get_selected_rows() != []


def test_update_row_replaces_existing() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw.update_row('/a', {'row_id': '/a', 'hierarchy_path': ['/a'], 'name': 'A2'})
    row = next(r for r in tw._rows if r['row_id'] == '/a')
    assert row['name'] == 'A2'


def test_replace_group_rows_updates_internal_state_without_grid() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw.replace_group_rows('/a', [{'row_id': '/a', 'hierarchy_path': ['/a'], 'name': 'A-updated'}])
    ids = {r['row_id'] for r in tw._rows}
    assert '/a' in ids
    assert '/a::1' not in ids


def test_get_displayed_rows_falls_back_to_python_rows_before_build() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    rows = asyncio.run(tw.get_displayed_rows())
    assert rows == _sample_rows()


def test_get_displayed_rows_uses_run_javascript_for_grid_state(monkeypatch: Any) -> None:
    scripts: list[str] = []

    async def fake_run_javascript(script: str, timeout: float = 1.0) -> list[dict[str, str]]:
        scripts.append(script)
        assert timeout == 5.0
        return [{'row_id': '/b'}, {'row_id': '/a'}]

    monkeypatch.setattr(tree_widget.ui, 'run_javascript', fake_run_javascript)
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw._grid = SimpleNamespace(id=77)

    rows = asyncio.run(tw.get_displayed_rows())

    assert rows == [{'row_id': '/b'}, {'row_id': '/a'}]
    assert len(scripts) == 1
    assert 'getElement(77)' in scripts[0]


def test_index_column_value_getter_uses_level_and_for_each_node() -> None:
    js = _index_column_value_getter_js()
    assert 'node.level !== 0' in js
    assert 'forEachNode' in js


def test_show_index_column_prepends_synthetic_column() -> None:
    """Index column uses AG Grid valueGetter; row data is not mutated."""
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(show_index_column=True),
    )
    assert tw._column_defs[0]['field'] == 'file_row_index'
    assert tw._column_defs[0]['headerName'] == ''
    assert tw._column_defs[0]['sortable'] is False
    assert ':valueGetter' in tw._column_defs[0]
    assert tw._column_defs[0][':valueGetter'] == _index_column_value_getter_js()
    assert tw._column_defs[0]['width'] == font_scaled_column_width_px(None)
    assert 'file_row_index' not in tw._rows[0]


def test_show_index_column_width_multiplier_scales_default_width() -> None:
    """Index column width should honor index_column_width_multiplier."""
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(show_index_column=True, index_column_width_multiplier=0.5),
    )
    base_width = font_scaled_column_width_px(None)
    assert tw._column_defs[0]['width'] == max(1, int(round(base_width * 0.5)))


def test_set_data_does_not_write_index_into_rows() -> None:
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(show_index_column=True),
    )
    tw.set_data(
        [
            {'row_id': '/z', 'hierarchy_path': ['/z'], 'name': 'Z'},
            {'row_id': '/z::1', 'hierarchy_path': ['/z', '/z::1'], 'name': 'Z1'},
        ]
    )
    for row in tw._rows:
        assert 'file_row_index' not in row


def test_set_data_updates_grid_row_data_only(monkeypatch) -> None:
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(show_index_column=True, clear_selection_on_set_data=False),
    )
    scripts: list[str] = []

    def _fake_run_javascript(script: str) -> None:
        scripts.append(script)

    monkeypatch.setattr(tree_widget.ui, 'run_javascript', _fake_run_javascript)
    tw._grid = SimpleNamespace(id=77, options={'rowData': []})  # type: ignore[assignment]
    tw._grid.update = lambda: None  # type: ignore[attr-defined]
    tw.set_data([{'row_id': '/z', 'hierarchy_path': ['/z'], 'name': 'Z'}])
    assert scripts == []
    assert tw._grid.options['rowData'] == [{'row_id': '/z', 'hierarchy_path': ['/z'], 'name': 'Z'}]


def test_set_data_builds_lazy_grid_when_first_rows_arrive(monkeypatch: Any) -> None:
    """First non-empty data must build the grid (born with rows), not update it.

    A grid created with empty ``rowData`` and later filled accepts programmatic
    ``setSelected`` state but never repaints the selected row. Building the grid
    only once rows exist guarantees it is born with rows and paints correctly.
    """
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=[],
        config=TreeWidgetConfig(clear_selection_on_set_data=False),
    )
    calls: list[str] = []
    monkeypatch.setattr(tw, '_ensure_grid_built', lambda: calls.append('build'))
    monkeypatch.setattr(tw, '_push_row_data_to_grid', lambda: calls.append('push'))

    tw.set_data(_sample_rows())

    assert calls == ['build']


def test_set_data_uses_update_when_grid_already_exists(monkeypatch: Any) -> None:
    """Subsequent data replacements keep using ``grid.update()`` (preserves state)."""
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(clear_selection_on_set_data=False),
    )
    calls: list[str] = []
    monkeypatch.setattr(tw, '_ensure_grid_built', lambda: calls.append('build'))
    monkeypatch.setattr(tw, '_push_row_data_to_grid', lambda: calls.append('push'))
    tw._grid = SimpleNamespace(id=42)  # type: ignore[assignment]

    tw.set_data([{'row_id': '/z', 'hierarchy_path': ['/z'], 'name': 'Z'}])

    assert calls == ['push']


def test_ensure_grid_built_no_op_without_root_or_rows() -> None:
    """The grid is created only once a root exists and rows are present."""
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=[])
    tw._ensure_grid_built()
    assert tw._grid is None
    tw._root = SimpleNamespace()  # type: ignore[assignment]
    tw._ensure_grid_built()
    assert tw._grid is None  # rows still empty


def test_replace_group_rows_builds_lazy_grid_when_absent(monkeypatch: Any) -> None:
    """First subtree of rows into a not-yet-built grid should build it lazily."""
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=[])
    calls: list[str] = []
    monkeypatch.setattr(tw, '_ensure_grid_built', lambda: calls.append('build'))

    tw.replace_group_rows('/a', [{'row_id': '/a', 'hierarchy_path': ['/a'], 'name': 'A'}])

    assert calls == ['build']
    assert [r['row_id'] for r in tw._rows] == ['/a']


def test_replace_group_rows_preserves_top_level_order() -> None:
    """Subtree replace keeps the group at its original rowData position."""
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(show_index_column=True),
    )
    tw.replace_group_rows('/a', [{'row_id': '/a', 'hierarchy_path': ['/a'], 'name': 'A-updated'}])
    top_level = [r for r in tw._rows if len(r['hierarchy_path']) == 1]
    assert [r['row_id'] for r in top_level] == ['/a', '/b']


def test_scroll_row_id_into_view_no_op_without_grid() -> None:
    """Scroll must be a safe no-op before the grid element is built."""
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw.scroll_row_id_into_view('/a')  # grid is None; must not raise


def test_scroll_row_id_into_view_expands_ancestors_and_scrolls_target() -> None:
    """Scroll must use documented AG Grid APIs on the grid's owning client."""
    scripts: list[str] = []
    client = SimpleNamespace(run_javascript=lambda script: scripts.append(script))
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw._grid = SimpleNamespace(id=55, client=client)  # type: ignore[assignment]

    tw.scroll_row_id_into_view('/a::1')

    assert len(scripts) == 1
    script = scripts[0]
    assert 'getElement(55)' in script
    assert 'getRowNode("/a::1")' in script
    assert 'const target =' in script
    assert 'setRowNodeExpanded' in script
    assert 'target,' in script
    assert '{forceSync: true}' in script
    assert "ensureNodeVisible(target, 'middle')" in script
    assert 'target.parent' not in script
    assert 'ancestor.setExpanded(true)' not in script
    assert 'requestAnimationFrame' not in script


def test_scroll_row_id_into_view_ignores_empty_id() -> None:
    """An empty row id must not trigger any client JavaScript."""
    scripts: list[str] = []
    client = SimpleNamespace(run_javascript=lambda script: scripts.append(script))
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw._grid = SimpleNamespace(id=55, client=client)  # type: ignore[assignment]

    tw.scroll_row_id_into_view('')

    assert scripts == []


def test_set_selected_row_ids_idempotent_skips_repeated_grid_churn() -> None:
    """Re-selecting the already-selected row must not re-issue grid commands.

    Repeated selection syncs of the same row (which happen per user click as
    lazy-load refreshes fire) previously produced a visible deselect/reselect
    flash. The idempotent guard makes the second identical sync a no-op.
    """
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    calls: list[tuple[str, tuple[Any, ...]]] = []
    tw._grid = SimpleNamespace(  # type: ignore[assignment]
        id=1,
        run_grid_method=lambda m, *a: calls.append(('grid', (m, *a))),
        run_row_method=lambda rid, m, *a: calls.append(('row', (rid, m, *a))),
    )

    tw.set_selected_row_ids(['/a'], origin='state')
    first = len(calls)
    tw.set_selected_row_ids(['/a'], origin='state')
    second = len(calls)

    assert calls == [('row', ('/a', 'setSelected', True, True))]
    assert first == 1
    assert second == first  # identical re-selection issues nothing
    # A genuinely different selection still issues commands.
    tw.set_selected_row_ids(['/b'], origin='state')
    assert len(calls) > second


def test_set_selected_row_ids_does_not_scroll(monkeypatch: Any) -> None:
    """Programmatic selection must NOT auto-scroll; scroll is a separate call.

    This guards the user-click path: a user clicking a tree row round-trips
    through ``set_selected_row_ids`` and must never trigger a scroll.
    """
    scripts: list[str] = []
    monkeypatch.setattr(tree_widget.ui, 'run_javascript', lambda s: scripts.append(s))
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    calls: list[str] = []
    monkeypatch.setattr(tw, 'scroll_row_id_into_view', lambda rid: calls.append(rid))
    tw._grid = SimpleNamespace(
        id=55,
        run_grid_method=lambda *a, **k: None,
        run_row_method=lambda *a, **k: None,
    )  # type: ignore[assignment]

    tw.set_selected_row_ids(['/a'], origin='state')

    assert calls == []


def test_show_index_column_false_omits_synthetic_column() -> None:
    tw = TreeWidget(
        columns=_sample_columns(),
        row_id_field='row_id',
        rows=_sample_rows(),
        config=TreeWidgetConfig(show_index_column=False),
    )
    fields = [c['field'] for c in tw._column_defs]
    assert 'file_row_index' not in fields
    assert 'file_row_index' not in tw._rows[0]


def test_show_index_column_rejects_conflicting_field() -> None:
    with pytest.raises(ValueError, match='conflicts with TreeWidgetConfig.index_field'):
        TreeWidget(
            columns=[ColumnDef(field='file_row_index', headerName='Dup'), *_sample_columns()],
            row_id_field='row_id',
            rows=_sample_rows(),
            config=TreeWidgetConfig(show_index_column=True),
        )


def _recording_grid() -> tuple[SimpleNamespace, list[tuple[str, tuple[Any, ...]]]]:
    calls: list[tuple[str, tuple[Any, ...]]] = []
    grid = SimpleNamespace(
        id=1,
        run_grid_method=lambda method, *args: calls.append(('grid', (method, *args))),
        run_row_method=lambda row_id, method, *args: calls.append(
            ('row', (row_id, method, *args))
        ),
    )
    return grid, calls


def test_single_row_selection_uses_one_atomic_grid_command() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    grid, calls = _recording_grid()
    tw._grid = grid  # type: ignore[assignment]

    tw.set_selected_row_ids(['/a'], origin='state')

    assert calls == [('row', ('/a', 'setSelected', True, True))]


def test_empty_selection_uses_deselect_all() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    grid, calls = _recording_grid()
    tw._grid = grid  # type: ignore[assignment]
    tw.set_selected_row_ids(['/a'])
    calls.clear()

    tw.set_selected_row_ids([])

    assert calls == [('grid', ('deselectAll',))]


def test_replace_group_rows_identical_data_sends_no_grid_commands() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    grid, calls = _recording_grid()
    tw._grid = grid  # type: ignore[assignment]

    tw.replace_group_rows('/a', _sample_rows()[:2])

    assert calls == []


def test_replace_group_rows_updates_only_changed_row() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    grid, calls = _recording_grid()
    tw._grid = grid  # type: ignore[assignment]
    replacement = [dict(row) for row in _sample_rows()[:2]]
    replacement[1]['name'] = 'A1-updated'

    tw.replace_group_rows('/a', replacement)

    assert calls == [
        (
            'grid',
            (
                'applyTransaction',
                {'update': [replacement[1]]},
            ),
        )
    ]


def test_replace_group_rows_adds_and_removes_only_changed_structure() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    grid, calls = _recording_grid()
    tw._grid = grid  # type: ignore[assignment]
    replacement = [
        {'row_id': '/a', 'hierarchy_path': ['/a'], 'name': 'A'},
        {'row_id': '/a::2', 'hierarchy_path': ['/a', '/a::2'], 'name': 'A2'},
    ]

    tw.replace_group_rows('/a', replacement)

    assert calls[0] == (
        'grid',
        (
            'applyTransaction',
            {
                'add': [replacement[1]],
                'remove': [{'row_id': '/a::1'}],
            },
        ),
    )
    assert calls[1] == ('row', ('/a', 'setExpanded', True))


def test_replace_group_rows_removing_selected_row_clears_tracking() -> None:
    tw = TreeWidget(columns=_sample_columns(), row_id_field='row_id', rows=_sample_rows())
    tw.set_selected_row_ids(['/a::1'])

    tw.replace_group_rows(
        '/a',
        [{'row_id': '/a', 'hierarchy_path': ['/a'], 'name': 'A'}],
    )

    assert tw.get_selected_rows() == []
    assert tw._selected_row_ids == []
    assert tw._last_selected_row_id is None
