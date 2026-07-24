"""Pure helper tests for NicePool control-panel behavior."""

from __future__ import annotations

from nicewidgets.nicepool.pool_control_panel import _default_plot_preset_dialog_name


def test_default_plot_preset_dialog_name_uses_selected_name() -> None:
    """Save dialog should default to the selected preset name when present."""
    assert _default_plot_preset_dialog_name(" Velocity plot ") == "Velocity plot"


def test_default_plot_preset_dialog_name_uses_fallback_for_empty_selection() -> None:
    """Save dialog should use a stable fallback when no preset is selected."""
    assert _default_plot_preset_dialog_name(None) == "my-plot"
    assert _default_plot_preset_dialog_name("") == "my-plot"
    assert _default_plot_preset_dialog_name("   ") == "my-plot"
