"""Top toolbar for plot pool app.

Provides build_toolbar() with Files select, Open, and Upload buttons.
"""

from __future__ import annotations

from typing import Callable

from nicegui import ui


def build_toolbar(
    *,
    csv_options: list[str],
    default_csv: str,
    on_csv_selected: Callable[[str], None],
    on_open_click: Callable[[], None],
    on_upload_click: Callable[[], None],
) -> ui.select:
    """Build top toolbar with Files select, Open, and Upload buttons.

    Args:
        csv_options: List of CSV filenames for the dropdown.
        default_csv: Initial selected filename.
        on_csv_selected: Callback when user selects a different CSV.
        on_open_click: Callback for Open button (placeholder for now).
        on_upload_click: Callback for Upload button (placeholder for now).

    Returns:
        The Files ui.select for optional programmatic updates.
    """
    def _on_change(e) -> None:
        val = getattr(e, "value", None)
        if isinstance(val, str) and val:
            on_csv_selected(val)

    with ui.row().classes("w-full items-center gap-3 flex-wrap"):
        options = csv_options if csv_options else ["(no CSV files)"]
        value = default_csv if csv_options and default_csv in csv_options else (csv_options[0] if csv_options else None)
        files_select = ui.select(
            options=options,
            value=value,
            on_change=_on_change,
        ).props('label="Files"').classes("min-w-[200px]")

        ui.button(
            icon="folder_open",
            on_click=on_open_click,
        ).props("flat dense").tooltip("Open CSV file")

        ui.button(
            icon="upload",
            on_click=on_upload_click,
        ).props("flat dense").tooltip("Upload CSV file")

    return files_select
