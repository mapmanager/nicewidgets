"""Reusable page builder for the TreeWidget demo."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from nicewidgets.aggrid_common.column_def import ColumnDef
from nicewidgets.tree_widget.config import TreeWidgetConfig
from nicewidgets.tree_widget.tree_widget import TreeWidget

try:
    from examples.tree_widget.sample_data import PATH_FIELD, ROW_ID_FIELD, make_demo_rows
except ImportError:
    from sample_data import PATH_FIELD, ROW_ID_FIELD, make_demo_rows  # type: ignore[no-redef]


def make_demo_column_defs() -> list[ColumnDef]:
    """Build demo columns (name is the tree group cell)."""
    return [
        ColumnDef(
            field='name',
            headerName='Name',
            extra={'cellRenderer': 'agGroupCellRenderer'},
        ),
        ColumnDef(field='kind', headerName='Kind'),
        ColumnDef(field='note', headerName='Note'),
    ]


def build_tree_demo_page(
    *,
    embedded: bool = False,
    dark_mode: bool = False,
) -> TreeWidget:
    """Build the TreeWidget demo in the current NiceGUI slot.

    Args:
        embedded: Whether the page is hosted inside another full-height layout.
        dark_mode: Initial AG Grid color scheme.

    Returns:
        Built tree widget.
    """
    height_class = 'h-full' if embedded else 'min-h-screen'
    with ui.column().classes(f'w-full {height_class} max-w-5xl mx-auto p-4 gap-4'):
        ui.label('TreeWidget demo').classes('text-h5')
        ui.label(
            'AG Grid Enterprise tree of experiments → files → ROIs. Click a row '
            'to select; use Expand/Collapse; right-click for a custom menu.'
        )
        ui.label(
            'AG Grid Enterprise requires a host-provided license in production.'
        ).classes('text-caption text-grey-7')
        selected = ui.label('Selected: (none)')

        def on_row_selected(row: dict[str, Any]) -> None:
            selected.text = (
                f"Selected: {row.get(ROW_ID_FIELD)!r} "
                f"({row.get('kind')}: {row.get('name')!r})"
            )

        def on_build_context_menu(tree: TreeWidget) -> None:
            ui.menu_item(
                'Select first ROI',
                on_click=lambda: tree.set_selected_row_ids(['roi-ctrl-0-1']),
            )
            ui.menu_item('Clear selection', on_click=tree.clear_selection)
            ui.separator()
            ui.menu_item('Expand all', on_click=tree.expand_all_nodes)
            ui.menu_item('Collapse all', on_click=tree.collapse_all_nodes)

        tree = TreeWidget(
            make_demo_column_defs(),
            ROW_ID_FIELD,
            make_demo_rows(),
            path_field=PATH_FIELD,
            on_row_selected=on_row_selected,
            on_build_context_menu=on_build_context_menu,
            config=TreeWidgetConfig(
                selection_mode='single',
                enable_keyboard_row_nav=True,
                show_index_column=True,
                index_header='Index',
            ),
        )
        with ui.column().classes('w-full').style('height: 420px;'):
            tree.build()
        tree.set_dark_mode(dark_mode)
        # Start expanded so the hierarchy is visible without an extra click.
        tree.expand_all_nodes()

        with ui.row().classes('gap-2 flex-wrap'):
            ui.button('Expand all', on_click=tree.expand_all_nodes)
            ui.button('Collapse all', on_click=tree.collapse_all_nodes)
            ui.button(
                'Select first ROI',
                on_click=lambda: tree.set_selected_row_ids(['roi-ctrl-0-1']),
            )
            ui.button('Clear selection', on_click=tree.clear_selection)

    return tree
