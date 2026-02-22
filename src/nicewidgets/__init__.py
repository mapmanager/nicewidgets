"""
nicewidgets: Reusable NiceGUI widgets for data visualization and interaction.

This package provides:
- CustomAgGrid: AG Grid wrapper with editable cells and row selection
- RoiImageWidget: Interactive image viewer with ROI drawing and zoom/pan
- Logging utilities for library and application use

For logging configuration in standalone scripts/demos:
    ```python
    from nicewidgets.utils.logging import configure_logging
    configure_logging(level="DEBUG")
    ```

When used as a library (imported by other applications), logging is
automatically handled by the parent application's configuration.
"""

import logging

from nicewidgets.utils.logging import configure_logging, get_logger

from nicewidgets.aggrid_gold_standard.gold_standard_aggrid_v2 import gold_standard_aggrid_v2
from nicewidgets.contrast_widget import ContrastParams, ContrastWidget

# Ensure nicewidgets logger has NullHandler so logs don't propagate to root
# when no application has configured logging. Applications/demos call
# configure_logging() to replace this with a real handler.
_logger = logging.getLogger("nicewidgets")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())

__all__ = [
    "ContrastParams",
    "ContrastWidget",
    "configure_logging",
    "get_logger",
    "gold_standard_aggrid_v2",
]

__version__ = "0.1.0"

