"""Tests for TableWidget resize and column-fit configuration."""

from __future__ import annotations

import sys
import types


if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.app = types.SimpleNamespace(native=False)
    fake_nicegui.events = types.SimpleNamespace(GenericEventArguments=object)
    fake_nicegui.ui = types.SimpleNamespace(
        on=lambda *_args, **_kwargs: None,
        column=lambda *_args, **_kwargs: None,
        context_menu=lambda *_args, **_kwargs: None,
        aggrid=lambda *_args, **_kwargs: None,
        menu_item=lambda *_args, **_kwargs: None,
        separator=lambda *_args, **_kwargs: None,
        notify=lambda *_args, **_kwargs: None,
    )
    sys.modules['nicegui'] = fake_nicegui

from nicewidgets.table_widget.column_def import ColumnDef
from nicewidgets.table_widget.config import TableWidgetConfig
from nicewidgets.table_widget.table_widget import TableWidget


def test_grid_size_change_handler_is_opt_in() -> None:
    """Default TableWidget config should not install resize-to-fit JS."""
    table = TableWidget(
        columns=(ColumnDef('name', 'Name'),),
        row_id_field='id',
        rows=({'id': 'row-1', 'name': 'A'},),
    )

    options = table._build_aggrid_options()

    assert ':onGridSizeChanged' not in options


def test_fit_columns_on_grid_resize_adds_grid_size_handler() -> None:
    """Opt-in TableWidget config should install AG Grid size-to-fit JS."""
    table = TableWidget(
        columns=(ColumnDef('name', 'Name'),),
        row_id_field='id',
        rows=({'id': 'row-1', 'name': 'A'},),
        config=TableWidgetConfig(fit_columns_on_grid_resize=True),
    )

    options = table._build_aggrid_options()

    assert options[':onGridSizeChanged'] == 'params => params.api.sizeColumnsToFit()'


def test_extra_grid_options_can_disable_default_filtering() -> None:
    """Caller-provided defaultColDef options should merge with defaults."""
    table = TableWidget(
        columns=(ColumnDef('name', 'Name'),),
        row_id_field='id',
        rows=({'id': 'row-1', 'name': 'A'},),
        config=TableWidgetConfig(
            extra_grid_options={
                'defaultColDef': {
                    'filter': False,
                },
            },
        ),
    )

    options = table._build_aggrid_options()

    assert options['defaultColDef']['sortable'] is True
    assert options['defaultColDef']['resizable'] is True
    assert options['defaultColDef']['filter'] is False
