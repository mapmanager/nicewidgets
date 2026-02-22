# Demo app for DataFrameTableView only (no LazySection, no PlotPoolController).
"""Demo application showing DataFrameTableView with a simple layout.

Loads the same CSV as demo_pool_app and displays it in a single aggrid table.
No expansion/lazy section — the table is inserted directly into the page.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import freeze_support
from typing import Any

import pandas as pd
from nicegui import ui

from nicewidgets.plot_pool_widget.dataframe_table_view import DataFrameTableView
from nicewidgets.utils import setUpGuiDefaults
from nicewidgets.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Configure logging once at import (like kymflow.gui_v2.app). Avoids duplicate
# handlers when the script is run in two processes (e.g. uv + NiceGUI native).
configure_logging(level="DEBUG")
import logging
print('xxx xxx', len(logging.getLogger().handlers))

def on_row_selected(row_dict: dict[str, Any]) -> None:
    logger.info("Row selected")
    from pprint import pprint
    pprint(row_dict, sort_dicts=False, indent=4)


def main() -> None:
    """Demo entrypoint: load CSV and show DataFrameTableView only."""

    path = "/Users/cudmore/Dropbox/data/declan/2026/compare-condiitons/v2-analysis/radon_report.csv"
    df = pd.read_csv(path)

    if "row_id" not in df.columns and "path" in df.columns and "roi_id" in df.columns:
        df["row_id"] = df["path"].astype(str) + "|" + df["roi_id"].astype(str)

    setUpGuiDefaults()
    ui.page_title("DataFrame Table Demo")

    with ui.column().classes("w-full h-full gap-4 p-4"):
        ui.label("DataFrame Table Demo").classes("text-2xl font-bold")
        wrapper = ui.column().classes("w-full").style("height: 500px;")
        table_view = DataFrameTableView(
            df,
            row_id_col="row_id" if "row_id" in df.columns else "path",
            on_row_selected=on_row_selected,
        )
        table_view.build(container=wrapper)

    ui.run(
        reload=False,
        native=True,
        window_size=(1000, 800),
    )


if __name__ in {"__main__", "__mp_main__"}:
    freeze_support()
    is_main_process = mp.current_process().name == "MainProcess"
    if is_main_process:
        print('=== === calling main()')
        main()
