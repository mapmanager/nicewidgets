# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/theme_v3.py
# v3-only theme installer.
# NOTE: absolute imports only (no relative imports).

from __future__ import annotations

from nicewidgets.custom_ag_grid_v3.styles_v3 import install_styles_v3

_installed = False


def install_theme_v3() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    install_styles_v3()