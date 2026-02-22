"""
Logging utilities for the nicewidgets library.

Library Logging Best Practices
-------------------------------
This module follows Python library logging conventions:

1. **Library code should NEVER call configure_logging()** - only use get_logger(__name__).
2. **Applications/examples CAN call configure_logging()** - to configure log output.
3. When imported by an application (e.g., kymflow) that has configured logging,
   all nicewidgets logs automatically use that application's handlers.

nicewidgets does NOT write any log files; it is a headless library.

Example Usage
-------------
In library code (roi_image_widget.py, grid.py, etc.):
    ```python
    from nicewidgets.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("ROI created")
    ```

In standalone examples/scripts:
    ```python
    from nicewidgets.utils.logging import configure_logging
    configure_logging(level="DEBUG")
    ```

When imported by kymflow (or any app with configured logging):
    - No need to call configure_logging()
    - All logs go to the application's configured handlers
    - No conflicts or duplicate messages
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Union

# Default format for nicewidgets logs
DEFAULT_FMT = "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d:%(funcName)s: %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: Optional[Union[str, int]] = None,
    *,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Configure logging for the nicewidgets logger only (never root).

    Use this in standalone scripts/demos that run ui.run() to enable log output.
    When nicewidgets is imported by an application (e.g., kymflow), that app
    configures logging; do not call this.

    nicewidgets never writes log files.

    Parameters
    ----------
    level:
        Logging level (e.g. "DEBUG", "INFO"). Defaults to NICEWIDGETS_LOG_LEVEL
        env var, or "INFO" if unset.
    fmt:
        Log message format. Defaults to a standard format.
    datefmt:
        Date format. Defaults to "%Y-%m-%d %H:%M:%S".
    force:
        If True, remove existing handlers before adding new ones (allows
        reconfiguration). If False, skip if handler already present.
    """
    if level is None:
        level = os.environ.get("NICEWIDGETS_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger("nicewidgets")
    logger.setLevel(level)

    if fmt is None:
        fmt = DEFAULT_FMT
    if datefmt is None:
        datefmt = DEFAULT_DATEFMT

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if force:
        for h in logger.handlers[:]:
            h.close()
            logger.removeHandler(h)
    else:
        # Skip if we already have a StreamHandler (e.g. from previous configure_logging)
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream is sys.stderr:
                return

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger by name.

    If name is None, returns a 'nicewidgets' logger.
    Otherwise, returns logging.getLogger(name).

    Use like:
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    if name is None:
        name = "nicewidgets"
    return logging.getLogger(name)
