"""
nicewidgets: Reusable NiceGUI widgets for data visualization and interaction.

This package provides:
- CustomAgGrid: AG Grid wrapper with editable cells and row selection
- RoiImageWidget: Interactive image viewer with ROI drawing and zoom/pan
- Logging utilities for library and application use

For logging configuration in standalone scripts:
    ```python
    from nicewidgets.utils.logging import setup_logging
    setup_logging(level="DEBUG", log_file="~/nicewidgets.log")
    ```

When used as a library (imported by other applications), logging is
automatically handled by the parent application's configuration.
"""

from nicewidgets.utils.logging import get_logger, get_log_file_path, setup_logging

from nicewidgets.aggrid_gold_standard.gold_standard_aggrid_v2 import gold_standard_aggrid_v2

__all__ = [
    "get_logger",
    "get_log_file_path",
    "gold_standard_aggrid_v2",
    "setup_logging",
]

__version__ = "0.1.0"

