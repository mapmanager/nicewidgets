"""Clipboard helpers for NiceGUI applications."""

from __future__ import annotations

import json
import logging
from io import BytesIO

try:
    import pyperclip
except ImportError:  # pragma: no cover - depends on optional runtime package
    pyperclip = None  # type: ignore[assignment]

from nicegui import ui

from nicewidgets.utils.desktop import is_pywebview_desktop
from nicewidgets.utils.logging import get_logger

logger = get_logger(__name__)


def copy_to_clipboard(text: str) -> None:
    """Copy text to the active system or browser clipboard.

    Native NiceGUI desktop windows use ``pyperclip`` so the operating-system
    clipboard is updated directly. Browser sessions use ``navigator.clipboard``
    through NiceGUI JavaScript execution.

    Args:
        text: Text to copy.

    Raises:
        RuntimeError: If the app is running in native mode and ``pyperclip`` is
            not installed.
    """
    if is_pywebview_desktop():
        if pyperclip is None:
            raise RuntimeError("pyperclip is required for native clipboard support")
        pyperclip.copy(text)
        logger.debug("copied text via pyperclip")
        return

    text_literal = json.dumps(text)
    ui.run_javascript(f"navigator.clipboard.writeText({text_literal});")
    logger.debug("copied text via browser clipboard")


def copy_png_bytes_to_native_clipboard(png_bytes: bytes) -> None:
    """Copy PNG image bytes to the native OS clipboard.

    Used by widget-level copy-to-clipboard actions when running inside a
    NiceGUI native window. Browser sessions cannot reliably write images to
    ``navigator.clipboard`` so widgets should call their browser-side
    PNG-to-clipboard helper instead.

    Args:
        png_bytes: PNG image bytes to copy.

    Raises:
        RuntimeError: If optional clipboard dependencies are not installed.
    """
    try:
        from PIL import Image
        import pyperclipimg as pci

        logging.getLogger("PIL").setLevel(logging.ERROR)
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependencies for image clipboard. Install pyperclipimg and pillow."
        ) from exc

    image = Image.open(BytesIO(png_bytes))
    pci.copy(image)
    logger.info("Copied PNG to native clipboard: %d bytes", len(png_bytes))
