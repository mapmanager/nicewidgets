"""Runtime detection helpers for NiceGUI desktop (pywebview) shells."""

from __future__ import annotations

from nicegui import app


def is_pywebview_desktop() -> bool:
    """Return whether the app is running inside a desktop pywebview shell.

    True for:

    * Legacy NiceGUI ``ui.run(native=True)`` (``app.native.main_window`` set)
    * Manual pywebview desktop shell (``webview.windows`` non-empty while
      ``ui.run(native=False, show=False)`` serves pages)

    False for normal browser sessions.

    Returns:
        ``True`` when clipboard, context-menu, and file-picker code should use
        desktop-specific behavior instead of browser APIs.
    """
    native_cfg = getattr(app, 'native', None)
    if getattr(native_cfg, 'main_window', None) is not None:
        return True
    try:
        import webview
    except ImportError:
        return False
    return len(webview.windows) > 0
