# Demo app for PlotPoolController
"""Demo application showing how to use PlotPoolController with and without LazySection.

This demo shows two usage patterns:
1. Direct usage: PlotPoolController.build() - renders immediately
2. Lazy usage: PlotPoolController.build_lazy() - renders only when expansion is opened
"""

from __future__ import annotations

import pandas as pd
from nicegui import ui


from nicewidgets.utils import setUpGuiDefaults
from nicewidgets.plot_pool_widget.plot_pool_controller import PlotPoolConfig, PlotPoolController
from nicewidgets.plot_pool_widget.lazy_section import LazySectionConfig

from nicewidgets.utils.logging import setup_logging

setup_logging(level="DEBUG")

# ----------------------------
# Demo entrypoint
# ----------------------------

def main() -> None:
    """Demo entrypoint showing both usage patterns for PlotPoolController.
    
    Pattern 1: Direct usage - renders immediately
    Pattern 2: Lazy usage - renders only when expansion is opened
    """

    # setup_logging(level="INFO")

    path = '/Users/cudmore/Dropbox/data/declan/2026/compare-condiitons/v2-analysis/radon_report.csv'

    def load_csv_and_build_df() -> pd.DataFrame:
        """Load the CSV and return a DataFrame with unique_row_id. Used by Refresh button."""
        df_new = pd.read_csv(path)
        if "row_id" not in df_new.columns and "path" in df_new.columns and "roi_id" in df_new.columns:
            df_new["unique_row_id"] = df_new["path"].astype(str) + "|" + df_new["roi_id"].astype(str)
        return df_new

    df = load_csv_and_build_df()

    setUpGuiDefaults('text-xs')
    
    ui.page_title("PlotPoolController Demo - Direct and Lazy Usage")
    
    with ui.column().classes("w-full gap-4 p-4"):

        cfg = PlotPoolConfig(
            pre_filter_columns=["roi_id"],
            unique_row_id_col="unique_row_id",
            on_refresh_requested=load_csv_and_build_df,
        )
        ctrl_lazy = PlotPoolController(df, config=cfg)
        ctrl_lazy.build_lazy(
            "Pool Plot (Lazy)",
            # subtitle="Click to load plot controls and visualization",
            config=LazySectionConfig(render_once=True, clear_on_close=False, show_spinner=True),
        )
        
        # Example: programmatically select a point by row_id
        # TODO: Replace with actual row_id value from your data
        # _select_path = '/Users/cudmore/Dropbox/data/declan/2026/compare-condiitons/v1-analysis/14d Saline/20251014/20251014_A98_0002.tif'
        # ctrl_direct.select_points_by_row_id(_select_path)
        # ctrl_lazy.select_points_by_row_id(_select_path)

    native_bool = True
    reload_bool = True
    if native_bool:
        ui.run(reload=reload_bool,
            native=True,
            window_size=(1000, 800)
            )
    else:
        ui.run(reload=reload_bool,
            native=False,
            # window_size=(1400, 800)
            )


if __name__ in {"__main__", "__mp_main__"}:
    main()
