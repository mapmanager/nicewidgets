"""Control panel UI for pool plotting application.

Builds and owns all left-panel widgets (layout, plot radio, filters, plot type,
group/color, X/Y column aggrids, plot options). Provides state↔widget sync
(bind_state, get_state) and sync_controls(plot_type).
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import pandas as pd
from nicegui import ui
from nicegui.events import GenericEventArguments

from nicewidgets.utils.logging import get_logger
from nicewidgets.nicepool.plot_state import PlotType, PlotState
from nicewidgets.nicepool.plot_helpers import (
    categorical_candidates,
    _ensure_aggrid_compact_css,
)
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE

logger = get_logger(__name__)

def _default_plot_preset_dialog_name(selected_name: object) -> str:
    """Return the initial name shown in the save-plot dialog.

    Args:
        selected_name: Current value from the saved-plot select widget.

    Returns:
        Existing selected name when present, otherwise ``"my-plot"``.
    """
    normalized = str(selected_name or "").strip()
    return normalized or "my-plot"


class PoolControlPanel:
    """Left-panel UI for plot configuration: layout, filters, plot type, X/Y columns, options."""

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        layout: str,
        current_plot_index: int,
        initial_state: PlotState,
        on_any_change: Callable[[], None],
        on_layout_change: Callable[[str], None],
        on_save_config: Callable[[], None],
        on_plot_radio_change: Callable[[Any], None],
        on_apply_current_to_others: Callable[[], None],
        on_replot_current: Callable[[], None],
        on_reset_to_default: Callable[[], None],
        on_copy_stats: Callable[[], None],
        on_copy_full_table: Callable[[], None],
        on_clear_selection: Callable[[], None] | None = None,
        on_x_column_selected: Callable[[dict[str, Any]], None],
        on_y_column_selected: Callable[[dict[str, Any]], None],
        show_save_button: bool = False,
        show_plot_presets: bool = True,
        get_plot_preset_names: Callable[[], list[str]] | None = None,
        on_plot_preset_selected: Callable[[str], bool] | None = None,
        on_save_plot_preset: Callable[[str], bool] | None = None,
        on_delete_plot_preset: Callable[[str], bool] | None = None,
    ) -> None:
        self.df = df
        self.layout = layout
        self.current_plot_index = current_plot_index
        self._initial_state = initial_state
        self._on_any_change = on_any_change
        self._on_layout_change = on_layout_change
        self._on_save_config = on_save_config
        self._on_plot_radio_change = on_plot_radio_change
        self._on_apply_current_to_others = on_apply_current_to_others
        self._on_replot_current = on_replot_current
        self._on_reset_to_default = on_reset_to_default
        self._on_copy_stats = on_copy_stats
        self._on_copy_full_table = on_copy_full_table
        self._on_clear_selection = on_clear_selection
        self._on_x_column_selected = on_x_column_selected
        self._on_y_column_selected = on_y_column_selected
        self._show_save_button = show_save_button
        self._show_plot_presets = show_plot_presets
        self._get_plot_preset_names = get_plot_preset_names
        self._on_plot_preset_selected = on_plot_preset_selected
        self._on_save_plot_preset = on_save_plot_preset
        self._on_delete_plot_preset = on_delete_plot_preset
        self._suppress_preset_select_event = False

        # Widget refs (set in build())
        self._layout_select: ui.select | None = None
        self._plot_radio: ui.radio | None = None
        self._pre_filter_selects: dict[str, ui.select] = {}
        self._type_select: ui.select | None = None
        self._x_aggrid: ui.aggrid | None = None
        self._y_aggrid: ui.aggrid | None = None
        self._group_select: ui.select | None = None
        self._color_grouping_select: ui.select | None = None
        self._ystat_select: ui.select | None = None
        self._cv_epsilon_input: ui.number | None = None
        self._histogram_bins_input: ui.number | None = None
        self._abs_value_checkbox: ui.checkbox | None = None
        self._swarm_jitter_amount_input: ui.number | None = None
        self._swarm_group_offset_input: ui.number | None = None
        self._use_remove_values_checkbox: ui.checkbox | None = None
        self._remove_values_threshold_input: ui.number | None = None
        self._show_mean_checkbox: ui.checkbox | None = None
        self._show_std_sem_checkbox: ui.checkbox | None = None
        self._std_sem_select: ui.select | None = None
        self._mean_line_width_input: ui.number | None = None
        self._error_line_width_input: ui.number | None = None
        self._show_raw_checkbox: ui.checkbox | None = None
        self._point_size_input: ui.number | None = None
        self._show_legend_checkbox: ui.checkbox | None = None
        self._preset_select: ui.select | None = None
        self._pending_timers: list[Any] = []

    def dispose(self) -> None:
        """Cancel deferred UI timers before this panel's container is destroyed."""
        for timer in self._pending_timers:
            try:
                timer.cancel()
            except Exception:
                pass
        self._pending_timers.clear()

    def build(
        self,
        pre_filter_options: dict[str, list[str]],
    ) -> None:
        """Build the control panel inside the current UI container.

        Creates layout select, plot radio, pre-filter dropdowns (one per key in
        pre_filter_options), plot type and options. Call once inside splitter.before.

        Args:
            pre_filter_options: Map column name -> list of option strings (e.g. ["(none)", "1", "2"]).
        """
        with ui.column().classes("w-full flex flex-col gap-4 p-4"):
            with ui.row().classes("w-full gap-2"):
                self._layout_select = ui.select(
                    options={"1x1": "1x1", "1x2": "1x2", "2x1": "2x1", "2x2": "2x2"},
                    value=self.layout,
                    label="Layout",
                    on_change=lambda e: self._on_layout_change(str(e.value)),
                ).classes("flex-1")
                if self._on_clear_selection is not None:
                    ui.button("Clear selection", on_click=self._on_clear_selection).classes("text-sm")
                if self._show_save_button:
                    ui.button("Save Config", on_click=self._on_save_config).classes("flex-1")

            with ui.row().classes("w-full gap-2 items-center"):
                ui.label("Edit Plot").classes("text-sm font-semibold")
                self._plot_radio = ui.radio(
                    ["1", "2", "3", "4"],
                    value=str(self.current_plot_index + 1),
                    on_change=self._on_plot_radio_change,
                ).props("inline")

            with ui.row().classes("w-full gap-2"):
                ui.button("Apply to Other", on_click=self._on_apply_current_to_others).classes("flex-1")
                ui.button("Replot", on_click=self._on_replot_current).classes("flex-1")
                ui.button("Reset Plots", on_click=self._on_reset_to_default).classes("flex-1")

            with ui.row().classes("w-full gap-2"):
                ui.button("Copy full table", on_click=self._on_copy_full_table).classes("flex-1")
                ui.button("Copy stats", on_click=self._on_copy_stats).classes("flex-1")

            self._build_saved_plots_section()

            with ui.card().classes("w-full"):
                ui.label("Pre Filter").classes("text-sm font-semibold")
                self._pre_filter_selects = {}
                for col, options in pre_filter_options.items():
                    initial_val = self._initial_state.pre_filter.get(col, PRE_FILTER_NONE)
                    initial_str = str(initial_val) if initial_val is not None else PRE_FILTER_NONE
                    if initial_str not in options and options:
                        initial_str = options[0]
                    self._pre_filter_selects[col] = ui.select(
                        options=options,
                        value=initial_str,
                        label=col,
                        on_change=self._on_any_change,
                    ).classes("w-full")
                self._abs_value_checkbox = ui.checkbox(
                    "Absolute Value",
                    value=self._initial_state.use_absolute_value,
                    on_change=self._on_any_change,
                ).classes("w-full")
                with ui.row().classes("w-full gap-2 items-center"):
                    self._use_remove_values_checkbox = ui.checkbox(
                        "Remove Values",
                        value=self._initial_state.use_remove_values,
                        on_change=self._on_any_change,
                    )
                    self._remove_values_threshold_input = ui.number(
                        label="Remove +/-",
                        value=self._initial_state.remove_values_threshold,
                        min=0.0,
                        step=0.1,
                        on_change=self._on_any_change,
                    ).classes("flex-1")

            _plot_type_labels = {
                "scatter": "Scatter",
                "swarm": "Swarm",
                "box_plot": "Box Plot",
                "violin": "Violin",
                "histogram": "Histogram",
                "cumulative_histogram": "Cumulative Histogram",
                "grouped": "Grouped",
            }
            self._type_select = ui.select(
                options={pt.value: _plot_type_labels.get(pt.value, pt.value) for pt in PlotType},
                value=self._initial_state.plot_type.value,
                label="Plot type",
                on_change=self._on_any_change,
            ).classes("w-full")

            group_options = ["(none)"] + categorical_candidates(self.df)
            self._group_select = ui.select(
                options=group_options,
                value=self._initial_state.group_col if self._initial_state.group_col else "(none)",
                label="Group/Color",
                on_change=self._on_any_change,
            ).classes("w-full")
            color_grouping_options = ["(none)"] + categorical_candidates(self.df)
            self._color_grouping_select = ui.select(
                options=color_grouping_options,
                value=self._initial_state.color_grouping if self._initial_state.color_grouping else "(none)",
                label="Group/Nesting",
                on_change=self._on_any_change,
            ).classes("w-full")
            self._ystat_select = ui.select(
                options=["mean", "median", "sum", "count", "std", "sem", "min", "max", "cv"],
                value=self._initial_state.ystat,
                label="Y stat (grouped)",
                on_change=self._on_any_change,
            ).classes("w-full")
            with ui.row().classes("w-full gap-2 items-center"):
                self._cv_epsilon_input = ui.number(
                    label="CV ε (|μ| < this → NaN)",
                    value=self._initial_state.cv_epsilon,
                    min=1e-20,
                    max=1.0,
                    step=1e-12,
                    on_change=self._on_any_change,
                ).classes("flex-1")
                self._histogram_bins_input = ui.number(
                    label="Histogram bins",
                    value=self._initial_state.histogram_bins,
                    min=5,
                    max=500,
                    step=5,
                    on_change=self._on_any_change,
                ).classes("flex-1")

            with ui.row().classes("w-full gap-2 items-center"):
                self._swarm_jitter_amount_input = ui.number(
                    label="Swarm Jitter",
                    value=self._initial_state.swarm_jitter_amount,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    on_change=self._on_any_change,
                ).classes("flex-1")
                self._swarm_group_offset_input = ui.number(
                    label="Swarm Offset",
                    value=self._initial_state.swarm_group_offset,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    on_change=self._on_any_change,
                ).classes("flex-1")

            with ui.row().classes("w-full gap-2 items-center"):
                self._show_mean_checkbox = ui.checkbox(
                    "Mean",
                    value=self._initial_state.show_mean,
                    on_change=self._on_any_change,
                )
                self._show_std_sem_checkbox = ui.checkbox(
                    "+/-",
                    value=self._initial_state.show_std_sem,
                    on_change=self._on_any_change,
                )
                self._std_sem_select = ui.select(
                    options=["std", "sem"],
                    value=self._initial_state.std_sem_type,
                    label="",
                    on_change=self._on_any_change,
                ).classes("flex-1")

            with ui.row().classes("w-full gap-2 items-start"):
                with ui.column().classes("flex-1"):
                    self._x_aggrid = self._create_column_aggrid(
                        "X column",
                        self._initial_state.xcol,
                        self._on_x_column_selected,
                    )
                with ui.column().classes("flex-1"):
                    self._y_aggrid = self._create_column_aggrid(
                        "Y column",
                        self._initial_state.ycol,
                        self._on_y_column_selected,
                    )

            self._build_plot_options()

    def _build_saved_plots_section(self) -> None:
        """Build named plot-preset controls when preset support is enabled."""
        if (
            not self._show_plot_presets
            or self._get_plot_preset_names is None
            or self._on_plot_preset_selected is None
            or self._on_save_plot_preset is None
            or self._on_delete_plot_preset is None
        ):
            return

        with ui.card().classes("w-full"):
            ui.label("Saved Plots").classes("text-sm font-semibold")
            self._preset_select = ui.select(
                options=self._get_plot_preset_names(),
                value=None,
                label="Plot",
                on_change=self._handle_plot_preset_selected,
            ).classes("w-full")
            with ui.row().classes("w-full gap-2 items-center"):
                ui.button("Save Plot", on_click=self._open_save_plot_dialog).classes("flex-1")
                ui.button("-", on_click=self._delete_selected_plot_preset).props('title="Delete selected plot"')

    def _handle_plot_preset_selected(self, event: GenericEventArguments) -> None:
        """Load a named preset selected by the user."""
        if self._suppress_preset_select_event:
            return
        if self._on_plot_preset_selected is None or event.value in (None, ""):
            return
        self._on_plot_preset_selected(str(event.value))

    def _delete_selected_plot_preset(self) -> None:
        """Delete the currently selected named preset without changing the plot."""
        if self._preset_select is None or self._on_delete_plot_preset is None:
            return
        name = str(self._preset_select.value or "").strip()
        if not name:
            ui.notify("No saved plot selected", type="warning")
            return
        deleted = self._on_delete_plot_preset(name)
        if deleted:
            self.refresh_plot_preset_options(selected_name=name)

    def _open_save_plot_dialog(self) -> None:
        """Open a small dialog for naming and saving the current plot preset."""
        if self._on_save_plot_preset is None:
            return

        default_name = _default_plot_preset_dialog_name(
            self._preset_select.value if self._preset_select is not None else None
        )
        with ui.dialog() as dialog, ui.card().classes("w-96"):
            ui.label("Save Plot").classes("text-lg font-semibold")
            name_input = ui.input("Plot name", value=default_name).classes("w-full")

            def save_and_close() -> None:
                name = str(name_input.value or "").strip()
                if not name:
                    ui.notify("Plot name cannot be empty", type="warning")
                    return
                saved = self._on_save_plot_preset(name)
                if saved:
                    self.refresh_plot_preset_options(selected_name=name)
                    dialog.close()

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dialog.close)
                ui.button("OK", on_click=save_and_close)

        dialog.open()

    def refresh_plot_preset_options(self, *, selected_name: str | None = None) -> None:
        """Refresh the saved-plot select options from the controller."""
        if self._preset_select is None or self._get_plot_preset_names is None:
            return
        names = self._get_plot_preset_names()
        self._suppress_preset_select_event = True
        try:
            self._preset_select.set_options(names)
            if selected_name in names:
                self._preset_select.value = selected_name
            self._preset_select.update()
        finally:
            self._suppress_preset_select_event = False

    def _build_plot_options(self) -> None:
        with ui.card().classes("w-full mt-4"):
            ui.label("Plot Options").classes("text-sm font-semibold")
            with ui.column().classes("w-full gap-3"):
                with ui.row().classes("w-full gap-2 items-center"):
                    self._mean_line_width_input = ui.number(
                        label="Mean Line Width",
                        value=self._initial_state.mean_line_width,
                        min=1,
                        max=10,
                        step=1,
                        on_change=self._on_any_change,
                    ).classes("flex-1")
                    self._error_line_width_input = ui.number(
                        label="Error Line Width",
                        value=self._initial_state.error_line_width,
                        min=1,
                        max=10,
                        step=1,
                        on_change=self._on_any_change,
                    ).classes("flex-1")
                with ui.row().classes("w-full gap-2 items-center"):
                    self._show_raw_checkbox = ui.checkbox(
                        "Raw",
                        value=self._initial_state.show_raw,
                        on_change=self._on_any_change,
                    )
                    self._point_size_input = ui.number(
                        label="Point Size",
                        value=self._initial_state.point_size,
                        min=1,
                        max=20,
                        step=1,
                        on_change=self._on_any_change,
                    ).classes("flex-1")
                self._show_legend_checkbox = ui.checkbox(
                    "Legend",
                    value=self._initial_state.show_legend,
                    on_change=self._on_any_change,
                ).classes("w-full")

    def _create_column_aggrid(
        self,
        label: str,
        initial_value: str,
        on_selected_callback: Callable[[dict[str, Any]], None],
    ) -> ui.aggrid:
        column_names = list(self.df.columns)
        row_data = [{"column": str(col)} for col in column_names]
        column_defs = [
            {
                "headerName": label,
                "field": "column",
                "sortable": True,
                "resizable": True,
                "flex": 2,
                "minWidth": 300,
            }
        ]
        grid_options: dict[str, Any] = {
            "columnDefs": column_defs,
            "rowData": row_data,
            "rowSelection": "single",
            "rowHeight": 22,
            "headerHeight": 26,
            "defaultColDef": {"sortable": True, "resizable": True},
            "autoSizeStrategy": {"type": "fitGridWidth"},
        }
        grid_options[":getRowId"] = "(params) => String(params.data.column)"
        _ensure_aggrid_compact_css()
        aggrid = ui.aggrid(grid_options).classes("w-full aggrid-compact")
        aggrid.style("height: 200px")

        async def on_row_selected(e: GenericEventArguments) -> None:
            try:
                selected_row = await aggrid.get_selected_row()
                if selected_row and selected_row.get("column"):
                    on_selected_callback(selected_row)
            except Exception as ex:
                logger.exception("Error handling %s row selection: %s", label, ex)

        aggrid.on("rowSelected", on_row_selected)
        if initial_value:
            def set_initial() -> None:
                try:
                    aggrid.run_row_method(initial_value, "setSelected", True, True)
                except RuntimeError:
                    pass
                except Exception:
                    pass

            timer = ui.timer(0.1, set_initial, once=True)
            self._pending_timers.append(timer)
        return aggrid

    def set_layout_and_current_plot(self, *, layout: str, current_plot_index: int) -> None:
        """Update layout and active-plot widgets without applying plot state."""
        self.layout = layout
        self.current_plot_index = current_plot_index
        if self._layout_select is not None:
            self._layout_select.value = layout
        if self._plot_radio is not None:
            self._plot_radio.value = str(current_plot_index + 1)

    def bind_state(self, state: PlotState) -> None:
        """Populate all widgets from a PlotState."""
        if not self._pre_filter_selects or not self._type_select or not self._x_aggrid or not self._y_aggrid:
            return
        for col, sel in self._pre_filter_selects.items():
            val = state.pre_filter.get(col, PRE_FILTER_NONE)
            sel.value = str(val) if val is not None else PRE_FILTER_NONE
        self._type_select.value = state.plot_type.value
        self._ystat_select.value = state.ystat
        if self._cv_epsilon_input is not None:
            self._cv_epsilon_input.value = state.cv_epsilon
        if self._histogram_bins_input is not None:
            self._histogram_bins_input.value = state.histogram_bins
        self._group_select.value = state.group_col if state.group_col else "(none)"
        self._color_grouping_select.value = state.color_grouping if state.color_grouping else "(none)"
        self._abs_value_checkbox.value = state.use_absolute_value
        self._swarm_jitter_amount_input.value = state.swarm_jitter_amount
        self._swarm_group_offset_input.value = state.swarm_group_offset
        self._use_remove_values_checkbox.value = state.use_remove_values
        self._remove_values_threshold_input.value = state.remove_values_threshold
        self._show_mean_checkbox.value = state.show_mean
        self._show_std_sem_checkbox.value = state.show_std_sem
        self._std_sem_select.value = state.std_sem_type
        self._mean_line_width_input.value = state.mean_line_width
        self._error_line_width_input.value = state.error_line_width
        self._show_raw_checkbox.value = state.show_raw
        self._point_size_input.value = state.point_size
        self._show_legend_checkbox.value = state.show_legend
        try:
            self._x_aggrid.run_row_method(state.xcol, "setSelected", True, True)
        except Exception:
            pass
        try:
            self._y_aggrid.run_row_method(state.ycol, "setSelected", True, True)
        except Exception:
            pass

    def get_state(self, *, xcol: str, ycol: str) -> PlotState:
        """Build PlotState from current widget values. xcol/ycol come from controller (aggrid async)."""
        pre_filter = {
            col: str(sel.value) if sel.value is not None else PRE_FILTER_NONE
            for col, sel in self._pre_filter_selects.items()
        }
        plot_type_str = str(self._type_select.value)
        try:
            plot_type = PlotType(plot_type_str)
        except ValueError:
            plot_type = PlotType.SCATTER
        gv = str(self._group_select.value)
        group_col = None if gv == "(none)" else gv
        cgv = str(self._color_grouping_select.value)
        color_grouping = None if cgv == "(none)" else cgv
        return PlotState(
            pre_filter=pre_filter,
            xcol=xcol,
            ycol=ycol,
            plot_type=plot_type,
            group_col=group_col,
            color_grouping=color_grouping,
            ystat=str(self._ystat_select.value),
            cv_epsilon=(
                float(self._cv_epsilon_input.value)
                if self._cv_epsilon_input is not None and self._cv_epsilon_input.value is not None
                else 1e-10
            ),
            histogram_bins=(
                int(self._histogram_bins_input.value)
                if self._histogram_bins_input is not None and self._histogram_bins_input.value is not None
                else 50
            ),
            use_absolute_value=bool(self._abs_value_checkbox.value),
            swarm_jitter_amount=float(self._swarm_jitter_amount_input.value or 0.35),
            swarm_group_offset=float(self._swarm_group_offset_input.value or 0.3),
            use_remove_values=bool(self._use_remove_values_checkbox.value),
            remove_values_threshold=(
                float(self._remove_values_threshold_input.value)
                if self._remove_values_threshold_input.value is not None
                else None
            ),
            show_mean=bool(self._show_mean_checkbox.value),
            show_std_sem=bool(self._show_std_sem_checkbox.value),
            std_sem_type=str(self._std_sem_select.value),
            mean_line_width=int(self._mean_line_width_input.value or 2),
            error_line_width=int(self._error_line_width_input.value or 2),
            show_raw=bool(self._show_raw_checkbox.value),
            point_size=int(self._point_size_input.value or 6),
            show_legend=bool(self._show_legend_checkbox.value),
        )

    def sync_controls(self, plot_type: PlotType, show_std_sem: bool = False) -> None:
        """Enable/disable controls based on plot type."""
        if not self._group_select or not self._ystat_select:
            return
        if self._cv_epsilon_input is not None:
            self._cv_epsilon_input.set_enabled(plot_type == PlotType.GROUPED)
        if self._histogram_bins_input is not None:
            self._histogram_bins_input.set_enabled(
                plot_type in {PlotType.HISTOGRAM, PlotType.CUMULATIVE_HISTOGRAM}
            )
        needs_group = plot_type in {
            PlotType.GROUPED,
            PlotType.SCATTER,
            PlotType.SWARM,
            PlotType.BOX_PLOT,
            PlotType.VIOLIN,
            PlotType.HISTOGRAM,
            PlotType.CUMULATIVE_HISTOGRAM,
        }
        is_grouped_agg = plot_type == PlotType.GROUPED
        show_mean_std = plot_type in {PlotType.SCATTER, PlotType.SWARM}
        supports_color_grouping = plot_type in {PlotType.BOX_PLOT, PlotType.VIOLIN, PlotType.SWARM, PlotType.SCATTER}
        self._group_select.set_enabled(needs_group)
        self._color_grouping_select.set_enabled(supports_color_grouping)
        self._ystat_select.set_enabled(is_grouped_agg)
        self._show_mean_checkbox.set_enabled(show_mean_std)
        self._show_std_sem_checkbox.set_enabled(show_mean_std)
        self._std_sem_select.set_enabled(show_mean_std and show_std_sem)
        self._swarm_jitter_amount_input.set_enabled(plot_type == PlotType.SWARM)
        self._swarm_group_offset_input.set_enabled(plot_type == PlotType.SWARM)
