"""Clipboard utility for NiceGUI apps.

Copy text to system clipboard. Supports native (pywebview) and browser modes.
"""

from __future__ import annotations

try:
    import pyperclip
except ImportError:
    pyperclip = None

from nicegui import app, ui

from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)


def copy_to_clipboard(text: str) -> None:
    """
    Copy text to system clipboard.

    Behavior:
    - If running NiceGUI in native=True (pywebview desktop app):
        Uses pyperclip to access OS clipboard directly.
    - If running in browser (native=False):
        Uses browser navigator.clipboard via JavaScript.
    - If pyperclip is unavailable in native mode:
        Raises RuntimeError.

    Args:
        text: Text to copy.

    Returns:
        None
    """
    native_cfg = getattr(app, "native", None)
    is_native_window = getattr(native_cfg, "main_window", None) is not None

    # print(text)
    if is_native_window:
        # Desktop mode
        if pyperclip is None:
            raise RuntimeError(
                "pyperclip not installed. Install it for native clipboard support."
            )
        pyperclip.copy(text)
        logger.debug("copied via pyperclip (native)")
    else:
        # Browser mode
        # Must escape backticks safely
        escaped = text.replace("\\", "\\\\").replace("`", "\\`")
        ui.run_javascript(f"""
            navigator.clipboard.writeText(`{escaped}`);
        """)
        logger.debug("copied via browser navigator.clipboard")
