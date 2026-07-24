from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SelectionMode = Literal['none', 'single', 'multiple']


def scaled_row_header_heights_px(cell_font_size_px: int) -> tuple[int, int]:
    """Return ``(row_height, header_height)`` in px from a cell font size.

    Used when callers want row/header chrome to track table font size without a
    separate persisted setting. Values are clamped for sensible AG Grid bounds.

    Args:
        cell_font_size_px: Body cell font size in pixels (typically ≥ 8).

    Returns:
        ``(row_height, header_height)`` both positive integers suitable for AG Grid
        ``rowHeight`` and ``headerHeight``.
    """
    fp = max(1, int(cell_font_size_px))
    row_h = max(28, min(64, fp * 2 + 12))
    header_h = max(28, min(56, fp + 24))
    return (row_h, header_h)


@dataclass(frozen=True, slots=True)
class TableWidgetConfig:
    """Grid-level options for ``TableWidget``.

    Attributes:
        selection_mode: Row selection behavior.
        clear_selection_on_set_data: Clear tracked/grid selection when replacing all rows.
        enable_edit_on_double_click: Start edit on double-click and emit edit-finished events.
        enable_keyboard_row_nav: ArrowUp/ArrowDown select previous/next displayed row.
        stop_editing_when_cells_lose_focus: End edits when focus leaves the grid.
        auto_size_columns: Forwarded to ``ui.aggrid(auto_size_columns=...)``.
        fit_columns_on_grid_resize: When true, AG Grid calls ``sizeColumnsToFit``
            after browser-side grid size changes. Defaults to false so existing
            tables keep their current resize behavior unless opted in.
        cell_font_size_px: When set, cell and header font size in pixels (merged into
            ``defaultColDef``). When ``None``, AG Grid theme defaults apply.
        row_height: Optional fixed row height (px) for AG Grid ``rowHeight``. When
            ``None``, the option is omitted (theme/browser default).
        header_height: Optional fixed header row height (px) for ``headerHeight``.
            When ``None``, the option is omitted.
        extra_grid_options: Additional AG Grid options merged before ``grid_options``.
        show_index_column: When true, prepend a synthetic 1-based ``Index`` column
            (stored in row data at ``index_field``); values follow ``rowData`` list
            order so they move with rows when the grid is sorted by other columns.
        index_field: Row dict / AG Grid field name for the index column (must not
            collide with application row keys).
        index_header: Column header label for the index column.
    """

    selection_mode: SelectionMode = 'single'
    clear_selection_on_set_data: bool = True
    enable_edit_on_double_click: bool = True
    enable_keyboard_row_nav: bool = True
    stop_editing_when_cells_lose_focus: bool = True
    auto_size_columns: bool = True
    fit_columns_on_grid_resize: bool = False
    cell_font_size_px: int | None = None
    row_height: int | None = None
    header_height: int | None = None
    extra_grid_options: dict[str, Any] = field(default_factory=dict)
    show_index_column: bool = True
    index_field: str = 'table_row_index'
    index_header: str = 'Index'
