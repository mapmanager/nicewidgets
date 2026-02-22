"""Theme utilities for Plotly charts.

Note: Single source of truth is kymflow.core.plotting.theme.
Sync or review when kymflow plotting changes.
"""

from __future__ import annotations

from enum import Enum


class ThemeMode(str, Enum):
    """UI theme mode.

    Used by plotting functions to coordinate theme settings.
    """

    DARK = "dark"
    LIGHT = "light"


def get_theme_colors(theme: ThemeMode) -> tuple[str, str]:
    """Get background and foreground colors for a theme."""
    if theme is ThemeMode.DARK:
        return "#000000", "#ffffff"
    return "#ffffff", "#000000"


def get_theme_template(theme: ThemeMode) -> str:
    """Get Plotly template name for a theme."""
    if theme is ThemeMode.DARK:
        return "plotly_dark"
    return "plotly_white"
