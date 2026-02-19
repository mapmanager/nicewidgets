"""Modular header component for plot pool app.

Provides build_plot_pool_header() with title, theme toggle, and GitHub link.
"""

from __future__ import annotations

import webbrowser

from nicegui import app, ui

THEME_STORAGE_KEY = "plot_pool_dark_mode"


def _open_external(url: str) -> None:
    """Open URL in system browser (native) or new tab (browser)."""
    native = getattr(app, "native", None)
    in_native = getattr(native, "main_window", None) is not None
    if in_native:
        webbrowser.open(url)
    else:
        ui.run_javascript(f'window.open("{url}", "_blank")')


def build_plot_pool_header(
    *,
    github_url: str = "https://github.com/mapmanager/nicewidgets",
) -> ui.dark_mode:
    """Build header with title, theme toggle, and GitHub link.

    Left: "Plot Pool" label.
    Right: theme toggle button, GitHub link.

    Args:
        github_url: URL for GitHub repository link.

    Returns:
        Dark mode controller for the page (for optional further use).
    """
    dark_mode = ui.dark_mode()
    stored = app.storage.user.get(THEME_STORAGE_KEY, True)
    dark_mode.value = stored

    def _update_theme_icon() -> None:
        icon = "light_mode" if dark_mode.value else "dark_mode"
        theme_btn.props(f"icon={icon}")

    def _toggle_theme() -> None:
        dark_mode.value = not dark_mode.value
        app.storage.user[THEME_STORAGE_KEY] = dark_mode.value
        _update_theme_icon()

    # with ui.header().classes("items-center justify-between"):

    with ui.header().classes("items-center justify-between").props("dense").style(
        "min-height: 36px; height: 36px; padding: 0 8px;"
    ):


        with ui.row().classes("items-center gap-2"):
            ui.label("Plot Pool").classes("!text-lg font-bold italic text-white")

        with ui.row().classes("items-center gap-2"):
            theme_btn = ui.button(
                icon="light_mode" if dark_mode.value else "dark_mode",
                on_click=_toggle_theme,
            ).props("flat round dense text-color=white").tooltip("Toggle dark / light mode")
            _update_theme_icon()

            github_icon = ui.image("https://cdn.simpleicons.org/github/ffffff").classes(
                "w-5 h-5 cursor-pointer"
            )
            github_icon.on("click", lambda: _open_external(github_url))
            github_icon.tooltip("Open GitHub repository")

    return dark_mode
