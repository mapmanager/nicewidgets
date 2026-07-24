"""Smoke tests for ``nicewidgets.table_widget`` (no live NiceGUI client)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nicewidgets.table_widget.column_def import ColumnDef
from nicewidgets.table_widget.config import TableWidgetConfig, scaled_row_header_heights_px
from nicewidgets.table_widget.js_hooks import (
    js_on_cell_double_clicked_start_editing,
    js_on_cell_editing_stopped_emit_change,
    js_on_cell_key_down_select_prev_next,
    js_on_row_clicked,
)
from nicewidgets.table_widget.table_widget import (
    TableWidget,
    _get_row_id_js_expression,
    validate_row_id_field,
    validate_rows_for_row_id_field,
)


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
        validate_rows_for_row_id_field(
            [
                {'id': 'x', 'v': 1},
                {'id': 'x', 'v': 2},
            ],
            'id',
        )


def test_column_def_extra_merged_into_aggrid_dict() -> None:
    c = ColumnDef(field='a', headerName='A', extra={'width': 120})
    d = c.as_aggrid_column_def()
    assert d['field'] == 'a'
    assert d['headerName'] == 'A'
    assert d['hide'] is False
    assert d['width'] == 120


def test_table_widget_config_defaults() -> None:
    cfg = TableWidgetConfig()
    assert cfg.selection_mode == 'single'
    assert cfg.clear_selection_on_set_data is True
    assert cfg.enable_edit_on_double_click is True
    assert cfg.enable_keyboard_row_nav is True
    assert cfg.show_index_column is True
    assert cfg.index_field == 'table_row_index'
    assert cfg.index_header == 'Index'
    assert cfg.cell_font_size_px is None
    assert cfg.row_height is None
    assert cfg.header_height is None


def test_scaled_row_header_heights_px_clamped() -> None:
    r, h = scaled_row_header_heights_px(13)
    assert r == 38 and h == 37
    r8, h8 = scaled_row_header_heights_px(8)
    assert r8 == 28
    r99, _ = scaled_row_header_heights_px(99)
    assert r99 == 64


def test_build_aggrid_options_applies_row_and_header_height() -> None:
    cols = (ColumnDef(field='id', headerName='ID'),)
    tw = TableWidget(
        cols,
        'id',
        rows=[{'id': 'a'}],
        config=TableWidgetConfig(row_height=36, header_height=34),
    )
    opts = tw._build_aggrid_options()
    assert opts['rowHeight'] == 36
    assert opts['headerHeight'] == 34


def test_build_aggrid_options_omits_row_header_height_when_none() -> None:
    cols = (ColumnDef(field='id', headerName='ID'),)
    tw = TableWidget(cols, 'id', rows=[{'id': 'a'}], config=TableWidgetConfig())
    opts = tw._build_aggrid_options()
    assert 'rowHeight' not in opts
    assert 'headerHeight' not in opts


def test_build_aggrid_options_applies_cell_font_size_px() -> None:
    cols = (ColumnDef(field='id', headerName='ID'),)
    tw = TableWidget(
        cols,
        'id',
        rows=[{'id': 'a'}],
        config=TableWidgetConfig(cell_font_size_px=11),
    )
    opts = tw._build_aggrid_options()
    assert opts['defaultColDef']['cellStyle']['fontSize'] == '11px'
    assert opts['defaultColDef']['headerStyle']['fontSize'] == '11px'


def test_build_aggrid_options_omits_font_when_cell_font_size_px_none() -> None:
    cols = (ColumnDef(field='id', headerName='ID'),)
    tw = TableWidget(cols, 'id', rows=[{'id': 'a'}], config=TableWidgetConfig())
    opts = tw._build_aggrid_options()
    assert 'cellStyle' not in opts['defaultColDef']
    assert 'headerStyle' not in opts['defaultColDef']


def test_build_aggrid_options_suppresses_enterprise_context_menu() -> None:
    """AG Grid Enterprise menu must be suppressed so only our menu shows."""
    cols = (ColumnDef(field='id', headerName='ID'),)
    tw = TableWidget(cols, 'id', rows=[{'id': 'a'}], config=TableWidgetConfig())
    opts = tw._build_aggrid_options()
    assert opts['suppressContextMenu'] is True
    assert opts['preventDefaultOnContextMenu'] is True


def test_get_row_id_js_expression_escapes_field_name() -> None:
    js = _get_row_id_js_expression('path')
    assert 'params.data' in js
    assert '"path"' in js


def test_js_hooks_include_emit_event() -> None:
    assert 'emitEvent' in js_on_row_clicked(emit_event='evt', row_id_field='id')
    assert 'ArrowUp' in js_on_cell_key_down_select_prev_next(emit_event='evt', row_id_field='id')
    assert 'startEditingCell' in js_on_cell_double_clicked_start_editing()
    assert 'oldValue' in js_on_cell_editing_stopped_emit_change(emit_event='evt', row_id_field='id')


def test_table_widget_upsert_replace_and_insert() -> None:
    tw = TableWidget(
        [ColumnDef(field='id', headerName='ID'), ColumnDef(field='v', headerName='V')],
        'id',
        [{'id': 'a', 'v': 1}],
    )
    tw.upsert_row({'id': 'a', 'v': 99})
    assert tw.get_rows() == [{'id': 'a', 'v': 99, 'table_row_index': 1}]
    tw.upsert_row({'id': 'b', 'v': 2})
    rows = tw.get_rows()
    assert len(rows) == 2
    assert {'id': 'a', 'v': 99, 'table_row_index': 1} in rows
    assert {'id': 'b', 'v': 2, 'table_row_index': 2} in rows


def test_table_widget_remove_row() -> None:
    tw = TableWidget(
        [ColumnDef(field='id', headerName='ID')],
        'id',
        [{'id': 'a'}, {'id': 'b'}],
    )
    tw.remove_row('a')
    assert tw.get_rows() == [{'id': 'b', 'table_row_index': 1}]
    with pytest.raises(ValueError, match='No row'):
        tw.remove_row('a')


def test_table_widget_update_row() -> None:
    tw = TableWidget([ColumnDef(field='id', headerName='ID'), ColumnDef(field='v', headerName='V')], 'id', [{'id': 'a', 'v': 1}])
    tw.update_row('a', {'id': 'a', 'v': 5})
    assert tw.get_rows() == [{'id': 'a', 'v': 5, 'table_row_index': 1}]


def test_programmatic_selection_single_mode() -> None:
    tw = TableWidget([ColumnDef(field='id', headerName='ID')], 'id', [{'id': 'a'}, {'id': 'b'}])
    tw.set_selected_row_ids(['a', 'b'])
    assert tw.get_selected_row_ids() == ['a']
    assert tw.get_selected_rows() == [{'id': 'a', 'table_row_index': 1}]
    tw.clear_selection()
    assert tw.get_selected_row_ids() == []


def test_programmatic_selection_none_mode() -> None:
    tw = TableWidget(
        [ColumnDef(field='id', headerName='ID')],
        'id',
        [{'id': 'a'}],
        config=TableWidgetConfig(selection_mode='none'),
    )
    tw.set_selected_row_ids(['a'])
    assert tw.get_selected_row_ids() == []


def test_set_data_clears_selection_by_default() -> None:
    tw = TableWidget([ColumnDef(field='id', headerName='ID')], 'id', [{'id': 'a'}])
    tw.set_selected_row_ids(['a'])
    assert tw.get_selected_row_ids() == ['a']
    tw.set_data([{'id': 'z'}])
    assert tw.get_selected_row_ids() == []
    assert tw.get_rows() == [{'id': 'z', 'table_row_index': 1}]


def test_set_data_keeps_selection_when_configured() -> None:
    tw = TableWidget(
        [ColumnDef(field='id', headerName='ID')],
        'id',
        [{'id': 'a'}],
        config=TableWidgetConfig(clear_selection_on_set_data=False),
    )
    tw.set_selected_row_ids(['a'])
    tw.set_data([{'id': 'z'}])
    # stale ids are kept with this opt-out behavior
    assert tw.get_selected_row_ids() == ['a']


def test_column_visibility_api_without_build() -> None:
    tw = TableWidget(
        [
            ColumnDef(field='a', headerName='A', hide=False),
            ColumnDef(field='b', headerName='B', hide=True),
        ],
        'id',
        [{'id': '1'}],
    )
    assert tw.get_column_visibility() == {'table_row_index': True, 'a': True, 'b': False}
    tw.set_column_visible('b', True)
    assert tw.get_column_visibility()['b'] is True
    tw.toggle_column_visible('a')
    assert tw.get_column_visibility()['a'] is False


def test_on_edit_emitted_updates_internal_rows_and_callback() -> None:
    called: list[tuple[str, str, Any, Any, dict[str, Any]]] = []

    def _cb(row_id: str, field: str, old: Any, new: Any, row: dict[str, Any]) -> None:
        called.append((row_id, field, old, new, row))

    tw = TableWidget(
        [ColumnDef(field='id', headerName='ID'), ColumnDef(field='kind', headerName='Kind', extra={'editable': True})],
        'id',
        [{'id': 'a', 'kind': 'tif'}],
        on_cell_edited=_cb,
    )
    tw._on_edit_emitted(
        SimpleNamespace(
            args={
                'rowId': 'a',
                'colId': 'kind',
                'oldValue': 'tif',
                'newValue': 'czi',
                'data': {'id': 'a', 'kind': 'czi', 'table_row_index': 1},
            }
        )
    )
    assert tw.get_rows()[0]['kind'] == 'czi'
    assert called and called[0][0] == 'a'


def test_show_index_column_false_omits_synthetic_column() -> None:
    tw = TableWidget(
        [ColumnDef(field='id', headerName='ID')],
        'id',
        [{'id': 'a'}],
        config=TableWidgetConfig(show_index_column=False),
    )
    assert tw.get_rows() == [{'id': 'a'}]
    assert tw.get_column_visibility() == {'id': True}
    assert tw._index_field is None


def test_index_field_conflict_with_column_raises() -> None:
    with pytest.raises(ValueError, match='conflicts'):
        TableWidget(
            [ColumnDef(field='table_row_index', headerName='Dup'), ColumnDef(field='id', headerName='ID')],
            'id',
            [{'id': 'a', 'table_row_index': 0}],
        )


def test_on_select_emitted_updates_selection_state_and_callback() -> None:
    got: list[dict[str, Any]] = []

    def _on_sel(row: dict[str, Any]) -> None:
        got.append(row)

    tw = TableWidget([ColumnDef(field='id', headerName='ID')], 'id', [{'id': 'a'}], on_row_selected=_on_sel)
    tw._on_select_emitted(
        SimpleNamespace(args={'rowId': 'a', 'rowIndex': 0, 'data': {'id': 'a', 'table_row_index': 1}})
    )
    assert tw.get_selected_row_ids() == ['a']
    assert got == [{'id': 'a', 'table_row_index': 1}]
