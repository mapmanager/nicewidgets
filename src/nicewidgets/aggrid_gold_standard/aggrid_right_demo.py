"""Demo app for gold_standard_aggrid_v2.

- Right-click on the grid → column toggle context menu
- Double-click on editable columns (grade, name) → in-cell select dropdown

Run: python -m nicewidgets.aggrid_gold_standard.aggrid_right_demo
"""

from __future__ import annotations

import pandas as pd
from nicegui import ui

from nicewidgets.aggrid_gold_standard.gold_standard_aggrid_v2 import gold_standard_aggrid_v2


def main() -> None:
    ui.label("gold_standard_aggrid_v2 demo").classes("text-lg font-semibold")
    ui.label(
        "Right-click → column toggle. Double-click grade/name → in-cell select dropdown."
    ).classes("text-sm opacity-80")

    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "score": [85, 92, 78, 88, 95],
            "grade": ["B", "A", "C", "B", "A"],
        }
    )

    with ui.column().classes("w-full").style("height: 400px;"):
        gold_standard_aggrid_v2(
            df,
            unique_row_id_col="id",
            editable_columns=["grade", "name"],
            row_select_callback=lambda row_id, row_dict: ui.notify(
                f"Selected: {row_dict.get('name')}"
            ),
        )

    ui.run(native=True, reload=False, window_size=(800, 600))


if __name__ in {"__main__", "__mp_main__"}:
    main()
