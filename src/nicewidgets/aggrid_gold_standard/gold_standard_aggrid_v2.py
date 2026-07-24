"""Compatibility AG Grid factory used by the faithful NicePool port."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from nicegui import ui


def gold_standard_aggrid_v2(
    df: pd.DataFrame,
    *,
    unique_row_id_col: str,
    row_select_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> ui.aggrid:
    """Build an AG Grid table from a DataFrame.

    Args:
        df: Source DataFrame.
        unique_row_id_col: Column containing stable row identifiers.
        row_select_callback: Optional callback invoked with row id and row data
            when a row is selected.

    Returns:
        NiceGUI AG Grid element.
    """
    rows = df.where(pd.notna(df), None).to_dict(orient="records")
    column_defs = [
        {
            "field": str(column),
            "headerName": str(column),
            "sortable": True,
            "filter": True,
            "resizable": True,
        }
        for column in df.columns
    ]
    grid = ui.aggrid(
        {
            "columnDefs": column_defs,
            "rowData": rows,
            "rowSelection": "single",
            "defaultColDef": {"sortable": True, "filter": True, "resizable": True},
            "domLayout": "normal",
        }
    ).classes("w-full h-full")

    def _on_selection(event: Any) -> None:
        args = getattr(event, "args", None)
        row: dict[str, Any] | None = None
        if isinstance(args, dict):
            selected = args.get("selectedRows")
            if isinstance(selected, list) and selected:
                row = selected[0]
            elif isinstance(args.get("data"), dict):
                row = args["data"]
        if row is None:
            return
        row_id = row.get(unique_row_id_col)
        if row_id is None:
            return
        if row_select_callback is not None:
            row_select_callback(str(row_id), dict(row))

    grid.on("selectionChanged", _on_selection)
    grid.on("rowSelected", _on_selection)
    return grid
