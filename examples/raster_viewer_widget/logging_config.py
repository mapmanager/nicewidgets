"""Logging configuration for the NiceGUI raster-viewer demo."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configure consistent console logging for the demo process.

    Existing root handlers are retained and updated so this function remains
    safe when NiceGUI or a test runner configured logging first.

    Args:
        level: Standard-library logging level applied to the root logger and its
            handlers.
    """
    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | "
            "%(funcName)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    for handler in root_logger.handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
    root_logger.setLevel(level)
