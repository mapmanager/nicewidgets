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
from nicewidgets.plot_pool_widget.plot_pool_controller import PlotPoolController
from nicewidgets.plot_pool_widget.lazy_section import LazySectionConfig

from nicewidgets.utils.logging import setup_logging


# ----------------------------
# Demo entrypoint
# ----------------------------

def main() -> None:
    """Demo entrypoint showing both usage patterns for PlotPoolController.
    
    Pattern 1: Direct usage - renders immediately
    Pattern 2: Lazy usage - renders only when expansion is opened
    """

    setup_logging(level="INFO")

    path = '/Users/cudmore/Dropbox/data/declan/2026/compare-condiitons/v2-analysis/radon_report.csv'

    df = pd.read_csv(path)
    if "row_id" not in df.columns and "path" in df.columns and "roi_id" in df.columns:
        df["unique_row_id"] = df["path"].astype(str) + "|" + df["roi_id"].astype(str)

    setUpGuiDefaults()
    
    ui.page_title("PlotPoolController Demo - Direct and Lazy Usage")
    
    with ui.column().classes("w-full gap-4 p-4"):

        # ui.label("PlotPoolController Usage Examples").classes("text-2xl font-bold")
        
        # # Example 1: Direct usage (renders immediately)
        # ui.label("Example 1: Direct Usage (renders immediately)").classes("text-lg font-semibold mt-4")
        # ui.label("The plot controller below renders immediately when the page loads.").classes("text-sm text-gray-600")
        
        # ctrl_direct = PlotPoolController(df, roi_id_col="roi_id", row_id_col="row_id", plot_state=None)
        # ctrl_direct.build()  # Renders immediately
        
        # ui.separator()
        
        # Example 2: Lazy usage (renders only when expansion is opened)
        # ui.label("Example 2: Lazy Usage (renders only when expansion is opened)").classes("text-lg font-semibold mt-4")
        # ui.label("The plot controller below uses build_lazy() - it only renders when you open the expansion.").classes("text-sm text-gray-600")
        
        ctrl_lazy = PlotPoolController(df, roi_id_col="roi_id", unique_row_id_col="unique_row_id", plot_state=None)
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
