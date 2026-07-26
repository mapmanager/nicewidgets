"""Public NicePool wrapper around the faithful plot-pool controller."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from nicegui import ui

from nicewidgets.nicepool.config import NicePoolConfig, resolve_pre_filter_columns
from nicewidgets.nicepool.plot_pool_controller import PlotPoolConfig, PlotPoolController, NICEPOOL_ROOT_CLASSES


class NicePool(PlotPoolController):
    """General-purpose DataFrame plotting and selection widget.

    ``NicePool`` preserves the original plot-pool GUI behavior while exposing a
    small stable API for host applications and scripts. It renders pre-filter
    dropdowns, plot-type controls, named presets, an optional data table, and
    one or more linked Plotly plots.

    DataFrame contract:
        - ``df`` must contain the column named by ``config.unique_row_id_col``
          (default ``"pool_row_id"``) with unique, non-empty string-able values.
          This column links table rows, plot points, and
          :meth:`select_points_by_row_ids`.
        - Categorical pre-filter columns are taken from
          ``config.pre_filter_columns`` when given, otherwise auto-detected from
          ``config.auto_pre_filter_columns`` (``accept``, ``channel``,
          ``roi_id``). Missing columns are ignored.
        - At least one numeric column is needed for the y-axis.

    See ``examples/nicepool`` for a runnable demo built on this contract.

    Args:
        df: Source DataFrame satisfying the contract above.
        config: Optional NicePool configuration. Defaults to ``NicePoolConfig()``.
        on_row_selected: Optional callback ``(row_id, row_dict) -> None`` invoked
            when a table row is selected. Overrides ``config.on_table_row_selected``.
        on_refresh_requested: Optional callback ``() -> pd.DataFrame`` invoked by
            the refresh button; its return value replaces the data. Overrides
            ``config.on_refresh_requested``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        config: NicePoolConfig | None = None,
        on_row_selected: Callable[[str, dict[str, object]], None] | None = None,
        on_refresh_requested: Callable[[], pd.DataFrame] | None = None,
    ) -> None:
        cfg = config if config is not None else NicePoolConfig()
        pre_filter_columns = resolve_pre_filter_columns(
            [str(column) for column in df.columns],
            explicit_columns=cfg.pre_filter_columns,
            auto_columns=cfg.auto_pre_filter_columns,
        )
        table_callback = on_row_selected if on_row_selected is not None else cfg.on_table_row_selected
        refresh_callback = on_refresh_requested if on_refresh_requested is not None else cfg.on_refresh_requested
        controller_config = PlotPoolConfig(
            pre_filter_columns=list(pre_filter_columns),
            unique_row_id_col=cfg.unique_row_id_col,
            db_type=cfg.db_type,
            app_name=cfg.app_name,
            config_path=cfg.config_path,
            plot_state=cfg.plot_state,
            initial_plot_config=cfg.initial_plot_config,
            on_table_row_selected=table_callback,
            on_refresh_requested=refresh_callback,
            show_save_button=cfg.show_save_button,
            show_selection_feedback=cfg.show_selection_feedback,
            show_table_widget=cfg.show_table_widget,
            enable_config_persistence=cfg.enable_config_persistence,
            dark_mode=cfg.dark_mode,
            enable_plot_presets=cfg.enable_plot_presets,
            plot_preset_path=cfg.plot_preset_path,
        )
        super().__init__(df, config=controller_config)
        self.nicepool_config = cfg
        self.pre_filter_columns = tuple(pre_filter_columns)

    def build(self, parent: ui.element | None = None, *, container: ui.element | None = None) -> ui.element:
        """Build the NicePool UI.

        Args:
            parent: Optional NiceGUI parent element.
            container: Optional legacy container argument.

        Returns:
            Root NiceGUI element containing the widget.
        """
        target = container if container is not None else parent
        if target is None:
            root = ui.column().classes(NICEPOOL_ROOT_CLASSES)
        else:
            with target:
                root = ui.column().classes(NICEPOOL_ROOT_CLASSES)
        super().build(container=root)
        return root

    def relayout_plots(self) -> None:
        """Rebuild Plotly figures after the widget container resizes.

        Returns:
            None.
        """
        super().relayout_plots()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        """Replace the source DataFrame and refresh the widget.

        Args:
            df: New source DataFrame.
        """
        self.update_df(df)

    def set_dark_mode(self, enabled: bool) -> None:
        """Set the Plotly layout theme from a dark-mode flag.

        Args:
            enabled: Whether dark mode is enabled.

        Returns:
            None.
        """
        super().set_dark_mode(enabled)

    def select_points_by_row_ids(
        self,
        row_ids: set[str] | list[str] | tuple[str, ...],
    ) -> None:
        """Programmatically select points matching any of the given row ids.

        Args:
            row_ids: Values from ``unique_row_id_col`` identifying rows to highlight.

        Returns:
            None.
        """
        super().select_points_by_row_ids(row_ids)
