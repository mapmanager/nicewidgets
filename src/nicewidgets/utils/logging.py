"""Logging utilities for NiceWidgets.

Applications should call :func:`setup_logging` once at startup.
Library/module code should call :func:`get_logger`.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

from platformdirs import user_config_dir

APP_NAME = 'nicewidgets'
DEFAULT_LOG_FILENAME = 'nicewidgets.log'
DISABLE_FILE_LOG_ENV = 'NICEWIDGETS_DISABLE_FILE_LOG'

_LOG_FILE_PATH: Path | None = None
_HANDLER_MARKER = '_nicewidgets_handler'


def setup_logging(
    level: str | int = 'INFO',
    *,
    app_name: str = APP_NAME,
    log_filename: str = DEFAULT_LOG_FILENAME,
    console: bool = True,
    file: bool = True,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> None:
    """Configure package logging for `nicewidgets`.

    Args:
        level: Console logging level as a string (e.g. ``"INFO"``) or int.
        app_name: Application name used by ``platformdirs`` for the log folder.
        log_filename: Name of the log file.
        console: Whether to emit logs to stderr.
        file: Whether to emit logs to a rotating log file.
        max_bytes: Max bytes per file before rotation.
        backup_count: Number of rotated files to keep.
    """
    resolved_level = _resolve_level(level)

    logger = logging.getLogger(APP_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    _remove_nicewidgets_handlers(logger)

    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(
            logging.Formatter('[%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s')
        )
        setattr(console_handler, _HANDLER_MARKER, True)
        logger.addHandler(console_handler)

    if file and not _file_log_disabled():
        global _LOG_FILE_PATH
        log_dir = Path(user_config_dir(app_name)) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        _LOG_FILE_PATH = log_dir / log_filename

        file_handler = logging.handlers.RotatingFileHandler(
            filename=_LOG_FILE_PATH,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8',
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            )
        )
        setattr(file_handler, _HANDLER_MARKER, True)
        logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger under the `nicewidgets` namespace."""
    return logging.getLogger(name or APP_NAME)


def get_log_file_path() -> Path | None:
    """Return the active log file path if configured."""
    return _LOG_FILE_PATH


def _resolve_level(level: str | int) -> int:
    """Resolve a logging level from string/int input."""
    if isinstance(level, int):
        return level
    if not isinstance(level, str):
        raise TypeError(f'Invalid level type: {type(level)}')
    level_upper = level.upper()
    try:
        return logging._nameToLevel[level_upper]
    except KeyError:
        raise ValueError(f'Unknown logging level: {level!r}') from None


def _remove_nicewidgets_handlers(logger: logging.Logger) -> None:
    """Remove handlers previously installed by :func:`setup_logging`."""
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()


def _file_log_disabled() -> bool:
    """Return whether file logging is disabled by environment variable."""
    value = os.getenv(DISABLE_FILE_LOG_ENV, '').strip().lower()
    return value in {'1', 'true', 'yes', 'on'}
