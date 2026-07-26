"""Configuration dataclass for :class:`nicewidgets.tree_widget.TreeWidget`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from nicewidgets.aggrid_common.enterprise import DEFAULT_AG_GRID_ENTERPRISE_MODULE_URL

SelectionMode = Literal['none', 'single', 'multiple']


@dataclass(frozen=True, slots=True)
class TreeWidgetConfig:
    """Grid-level options for ``TreeWidget``.

    Mirrors the relevant fields of
    :class:`nicewidgets.table_widget.config.TableWidgetConfig`. Inline cell
    editing is not part of the v1 surface.

    Attributes:
        selection_mode: Row selection behavior.
        clear_selection_on_set_data: Clear tracked/grid selection when
            replacing all rows via :meth:`TreeWidget.set_data`.
        enable_keyboard_row_nav: ArrowUp/ArrowDown select previous/next
            displayed row.
        auto_size_columns: Forwarded to ``ui.aggrid(auto_size_columns=...)``.
        fit_columns_on_grid_resize: When true, AG Grid calls
            ``sizeColumnsToFit`` after browser-side grid size changes.
        suppress_movable_columns: When true, users cannot drag columns to
            reorder them.
        cell_font_size_px: When set, cell and header font size in pixels
            (merged into ``defaultColDef``). When ``None``, AG Grid theme
            defaults apply.
        row_height: Optional fixed row height (px). When ``None``, the option
            is omitted (theme/browser default).
        header_height: Optional fixed header row height (px). When ``None``,
            the option is omitted.
        extra_grid_options: Additional AG Grid options merged before any
            ``grid_options`` constructor argument.
        enterprise_module_url: AG Grid Enterprise ESM module URL passed to
            ``ui.aggrid.set_module_source``. ``set_module_source`` is invoked
            once at widget construction time. When ``None``, no override is
            applied (caller is assumed to have configured the bundle).
        show_index_column: When true, prepend a synthetic 1-based Index column
            for top-level tree rows only (``node.level === 0`` in AG Grid).
            Child rows are blank. Indices follow row-model load order via a
            client-side ``valueGetter`` (not stored in row data).
        index_field: AG Grid column field name for the index column (must not
            collide with application row keys).
        index_header: Column header label for the index column. Use ``''`` for
            a blank header (column remains visible).
        index_menu_label: Label used in the column visibility context menu
            when ``index_header`` is blank (default ``Index``).
        index_column_width_multiplier: Scale factor applied to the default
            font-scaled index column width (``1.0`` keeps the default).
    """

    selection_mode: SelectionMode = 'single'
    clear_selection_on_set_data: bool = True
    enable_keyboard_row_nav: bool = True
    auto_size_columns: bool = True
    fit_columns_on_grid_resize: bool = False
    suppress_movable_columns: bool = False
    cell_font_size_px: int | None = None
    row_height: int | None = None
    header_height: int | None = None
    extra_grid_options: dict[str, Any] = field(default_factory=dict)
    enterprise_module_url: str | None = DEFAULT_AG_GRID_ENTERPRISE_MODULE_URL
    show_index_column: bool = False
    index_field: str = 'file_row_index'
    index_header: str = ''
    index_menu_label: str = 'Index'
    index_column_width_multiplier: float = 1.0


def scaled_row_header_heights_px(cell_font_size_px: int) -> tuple[int, int]:
    """Return ``(row_height, header_height)`` in px from a cell font size.

    Mirrors ``nicewidgets.table_widget.config.scaled_row_header_heights_px``
    so tree-view callers can reuse the same row/header chrome scaling without
    crossing the table-widget package boundary.

    Args:
        cell_font_size_px: Body cell font size in pixels (typically >= 8).

    Returns:
        ``(row_height, header_height)`` both positive integers suitable for
        AG Grid ``rowHeight`` and ``headerHeight``.
    """
    fp = max(1, int(cell_font_size_px))
    row_h = max(28, min(64, fp * 2 + 12))
    header_h = max(28, min(56, fp + 24))
    return (row_h, header_h)


def font_scaled_column_width_px(
    cell_font_size_px: int | None,
    *,
    multiplier: int = 6,
    minimum: int = 36,
) -> int:
    """Return AG Grid column width from body font size.

    Args:
        cell_font_size_px: Body cell font size in pixels, or ``None`` for 13px.
        multiplier: Width multiplier applied to font size.
        minimum: Minimum column width in pixels.

    Returns:
        Column width in pixels suitable for AG Grid ``width`` / ``minWidth``.
    """
    fp = max(1, int(cell_font_size_px if cell_font_size_px is not None else 13))
    return max(minimum, int(fp * multiplier))
