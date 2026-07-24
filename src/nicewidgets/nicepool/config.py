"""Public configuration helpers for :mod:`nicewidgets.nicepool`.

The faithful NicePool implementation is a DataFrame-driven plotting widget. This
module provides the public configuration name while preserving the underlying
plot-pool configuration fields from the reference implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from nicewidgets.nicepool.plot_pool_controller import PlotPoolConfig
from nicewidgets.nicepool.plot_state import PlotState

DEFAULT_AUTO_PRE_FILTER_COLUMNS: tuple[str, ...] = ("accept", "channel", "roi_id")


@dataclass
class NicePoolConfig(PlotPoolConfig):
    """Configuration for the public ``NicePool`` widget.

    Args:
        pre_filter_columns: Explicit categorical columns to expose as
            pre-filter controls. Missing columns are ignored by the widget.
        unique_row_id_col: Column containing stable row identifiers.
        db_type: Logical dataframe type used to scope optional saved plot
            configuration.
        app_name: Optional application name for optional configuration
            persistence.
        config_path: Optional explicit configuration path used when persistence
            is enabled.
        plot_state: Optional fallback plot state when no startup config applies.
        initial_plot_config: Optional inline plot config dict (layout + plot_states).
            Takes precedence over session persistence when set.
        on_table_row_selected: Optional row-selection callback used by the
            underlying table view.
        on_refresh_requested: Optional callback used by the refresh button.
        show_save_button: Whether to render the save-config button.
        show_selection_feedback: Whether to render the selection feedback row.
        show_table_widget: Whether to render the optional DataFrame table.
        auto_pre_filter_columns: Candidate columns used when
            ``pre_filter_columns`` is ``None``.
        table_font_size_px: Reserved for future table style integration.
        enable_config_persistence: Whether to load/save plot configuration.
        dark_mode: Initial Plotly layout theme for generated figures.
        enable_plot_presets: Whether to show and persist named plot presets.
        plot_preset_path: Optional explicit path for named plot presets.
    """

    pre_filter_columns: Sequence[str] | None = None
    unique_row_id_col: str = "pool_row_id"
    db_type: str = "default"
    app_name: str | None = None
    config_path: Path | None = None
    plot_state: PlotState | None = None
    initial_plot_config: dict[str, Any] | None = None
    on_table_row_selected: Callable[[str, dict[str, object]], None] | None = None
    on_refresh_requested: Callable[[], pd.DataFrame] | None = None
    show_save_button: bool = False
    show_selection_feedback: bool = False
    show_table_widget: bool = False
    auto_pre_filter_columns: Sequence[str] = field(default_factory=lambda: DEFAULT_AUTO_PRE_FILTER_COLUMNS)
    table_font_size_px: int | None = None
    enable_config_persistence: bool = False
    dark_mode: bool = False
    enable_plot_presets: bool = True
    plot_preset_path: Path | None = None


def resolve_pre_filter_columns(
    available_columns: Sequence[str],
    *,
    explicit_columns: Sequence[str] | None = None,
    auto_columns: Sequence[str] = DEFAULT_AUTO_PRE_FILTER_COLUMNS,
) -> tuple[str, ...]:
    """Return pre-filter columns present in a DataFrame schema.

    Args:
        available_columns: DataFrame column names.
        explicit_columns: Caller-provided columns. When omitted, conventional
            columns are auto-detected.
        auto_columns: Candidate columns for auto-detection.

    Returns:
        Tuple of column names that exist in ``available_columns``.
    """
    available = {str(column) for column in available_columns}
    candidates = explicit_columns if explicit_columns is not None else auto_columns
    return tuple(str(column) for column in candidates if str(column) in available)
