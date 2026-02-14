"""Plot controller for pool plotting application.

Provides PlotPoolController, the main entry point for building interactive pool
plot UIs with NiceGUI: data table, pre-filter dropdowns, controls, and Plotly
plots with linked selection. Supports build() (immediate) and build_lazy() (render
when expansion is opened). See PlotPoolController class docstring for public API.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd
from nicegui import ui
from nicegui.events import GenericEventArguments

from nicewidgets.utils.logging import get_logger
from nicewidgets.plot_pool_widget.plot_state import PlotType, PlotState
from nicewidgets.plot_pool_widget.plot_helpers import numeric_columns, is_categorical_column
from nicewidgets.plot_pool_widget.dataframe_processor import DataFrameProcessor
from nicewidgets.plot_pool_widget.figure_generator import FigureGenerator
from nicewidgets.plot_pool_widget.pool_control_panel import PoolControlPanel
from nicewidgets.plot_pool_widget.selection_handler import PlotSelectionHandler, is_selection_compatible
from nicewidgets.plot_pool_widget.lazy_section import LazySection, LazySectionConfig
from nicewidgets.plot_pool_widget.dataframe_table_view import DataFrameTableView
from nicewidgets.plot_pool_widget.pre_filter_conventions import (
    PRE_FILTER_NONE,
    default_pre_filter,
    format_pre_filter_display,
)

logger = get_logger(__name__)


class PlotPoolController:
    """Controller for interactive pool plotting with NiceGUI.

    Manages plot state, UI widgets, and user interactions for creating
    interactive Plotly visualizations with linked selection across multiple plots.
    Loads/saves layout and plot state via PoolPlotConfig (schema v3).

    **Public API (entry point for external users):**

    - **__init__(df, ...)** — Configure with dataframe, pre-filter columns, row-id column, and optional callbacks.
    - **build(container=None)** — Build the full UI (header, table, controls, plots). Call once to render.
    - **build_lazy(title, ...)** — Build the same UI inside a LazySection (renders when expansion is opened).
    - **select_points_by_row_id(row_id)** — Programmatically select points in the plots that match the given row_id.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        pre_filter_columns: Optional[list[str]] = None,
        unique_row_id_col: str = "path",
        plot_state: Optional[PlotState] = None,
        on_table_row_selected: Optional[Callable[[str, dict[str, Any]], None]] = None,
        on_refresh_requested: Optional[Callable[[], pd.DataFrame]] = None,
    ) -> None:
        """Initialize plot controller with dataframe and column configuration.

        Args:
            df: DataFrame containing plot data. Must have numeric columns for x/y,
                and columns for pre_filter_columns and unique_row_id_col.
            pre_filter_columns: Column names used for pre-filter dropdowns (one dropdown per column).
                Each column's unique values are shown as options; selection filters the data before plotting.
                Default is ["roi_id"]. Pass a list of categorical column names, e.g. ["roi_id"] or ["condition", "batch"].
            unique_row_id_col: Column name that uniquely identifies each row (e.g. "path", "id").
                Used for table selection, plot click-to-row mapping, and select_points_by_row_id().
            plot_state: Optional initial PlotState. If None, defaults are used (first pre-filter value, first numeric x/y).
            on_table_row_selected: Optional callback invoked when the user selects a row in the data table.
                Signature: (row_id: str, row_dict: dict[str, Any]) -> None.
                - row_id: The value in the selected row for unique_row_id_col (as string).
                - row_dict: The full row as a dict (column name -> value).
                The controller also clears plot selection and highlights the corresponding point(s) in the plots.
            on_refresh_requested: Optional callback invoked when the user clicks the Refresh button.
                Signature: () -> pd.DataFrame. Should load fresh data and return a DataFrame with
                unique_row_id_col, pre_filter_columns, and numeric columns. If set, a Refresh button
                is shown in the header; on click, the callback is called and update_df() is invoked.
        """
        self.df = df
        self.pre_filter_columns = pre_filter_columns if pre_filter_columns is not None else ["roi_id"]
        self.unique_row_id_col = unique_row_id_col
        self._on_table_row_selected = on_table_row_selected
        self._on_refresh_requested = on_refresh_requested

        # Initialize DataFrameProcessor for data operations
        self.data_processor = DataFrameProcessor(
            df,
            pre_filter_columns=self.pre_filter_columns,
            unique_row_id_col=unique_row_id_col,
        )

        # Initialize FigureGenerator for plot generation
        self.figure_generator = FigureGenerator(
            self.data_processor,
            unique_row_id_col=unique_row_id_col,
        )

        # reasonable defaults
        num_cols = numeric_columns(df)
        if not num_cols:
            raise ValueError("Need at least one numeric column for y.")
        x_default = num_cols[0]
        y_default = num_cols[1] if len(num_cols) >= 2 else num_cols[0]
        
        # Default pre_filter: first column's first value, rest (none); or all (none)
        initial_pre_filter = default_pre_filter(self.pre_filter_columns)
        if self.pre_filter_columns:
            first_vals = self.data_processor.get_pre_filter_values(self.pre_filter_columns[0])
            if first_vals:
                initial_pre_filter = dict(initial_pre_filter)
                initial_pre_filter[self.pre_filter_columns[0]] = first_vals[0]

        # Initialize with 2 plot states (extensible to 4)
        if plot_state is None:
            default_state = PlotState(
                pre_filter=initial_pre_filter,
                xcol=x_default,
                ycol=y_default,
            )
        else:
            default_state = plot_state
        
        # Store default plot state for reset functionality
        self.default_plot_state: PlotState = PlotState.from_dict(default_state.to_dict())
        
        # Try to load saved config, otherwise use provided/default plot_state
        from nicewidgets.plot_pool_widget.pool_plot_config import PoolPlotConfig
        config = PoolPlotConfig.load()
        loaded_plot_states = config.get_plot_states()
        loaded_layout = config.get_layout()
        self._control_panel_splitter_value: float = config.get_control_panel_splitter_value()
        self._splitter_save_timer: Optional[Any] = None

        if loaded_plot_states:
            logger.info(f"Loaded {len(loaded_plot_states)} plot state(s) and layout '{loaded_layout}' from pool_plot_config.json")
            # Use loaded layout
            self.layout = loaded_layout
            # Use loaded plot states, pad with default_state if needed
            num_plots_needed = self._get_num_plots_for_layout(loaded_layout)
            self.plot_states = []
            for i in range(num_plots_needed):
                if i < len(loaded_plot_states):
                    self.plot_states.append(loaded_plot_states[i])
                else:
                    self.plot_states.append(PlotState.from_dict(default_state.to_dict()))
            # Update stored default to match first loaded state
            self.default_plot_state = PlotState.from_dict(self.plot_states[0].to_dict())
        else:
            logger.info("No saved plot config found, using provided/default plot state")
            self.layout = "1x1"
            # Initialize with 4 plot states to support 2x2 layout
            self.plot_states = [
                PlotState.from_dict(default_state.to_dict()),
                PlotState.from_dict(default_state.to_dict()),
                PlotState.from_dict(default_state.to_dict()),
                PlotState.from_dict(default_state.to_dict()),
            ]
        
        self.current_plot_index: int = 0

        # UI handles
        self._plots: list[ui.plotly] = []
        self._clicked_label: Optional[ui.label] = None
        self._mainSplitter: Optional[ui.splitter] = None
        self._verticalSplitter: Optional[ui.splitter] = None
        self._last_plot_type: Optional[PlotType] = None
        self._plot_container: Optional[ui.column] = None
        self._selection_label: Optional[ui.label] = None
        self._control_panel: Optional[PoolControlPanel] = None
        self._table_view: Optional[DataFrameTableView] = None

        # row_id -> iloc index within CURRENT filtered df (rebuilt each replot)
        self._id_to_index_filtered: dict[str, int] = {}

        # Linked selection (owned by handler; controller uses get_selected_row_ids())
        self._selection_handler = PlotSelectionHandler(
            data_processor=self.data_processor,
            figure_generator=self.figure_generator,
            unique_row_id_col=self.unique_row_id_col,
            get_filtered_df=self._get_filtered_df,
            on_apply_selection=self._apply_selection_to_all_plots,
            on_update_label=self._set_selection_label_count,
        )

    def select_points_by_row_id(self, row_id: str) -> None:
        """Programmatically select points in the plots that match the given row_id (public API).

        Finds the row_id in the current filtered data (per plot state), highlights
        the corresponding point(s) in scatter/swarm plots, and updates the selection label.
        Use this after table selection or external logic to sync plot selection.

        Args:
            row_id: Value from unique_row_id_col that identifies the row (e.g. path or id as string).
        """
        self._selection_handler.select_by_row_id(row_id, self.plot_states)

    def update_df(self, new_df: pd.DataFrame) -> None:
        """Replace the dataframe and refresh table, plots, and controls.

        Call this when the underlying data has changed. Rebuilds the table,
        control panel, and plots with the new data. No-op if the controller
        has not been built yet (e.g., LazySection closed).

        Args:
            new_df: New DataFrame. Must have unique_row_id_col, pre_filter_columns,
                and at least one numeric column for y.
        """
        if self._verticalSplitter is None or self._table_container is None or self._control_panel_container is None:
            return
        if self.unique_row_id_col not in new_df.columns:
            raise ValueError(f"new_df must have column {self.unique_row_id_col!r}")
        for col in self.pre_filter_columns:
            if col not in new_df.columns:
                raise ValueError(f"new_df must have column {col!r}")
        num_cols = numeric_columns(new_df)
        if not num_cols:
            raise ValueError("new_df must have at least one numeric column")

        self.df = new_df
        self.data_processor = DataFrameProcessor(
            new_df,
            pre_filter_columns=self.pre_filter_columns,
            unique_row_id_col=self.unique_row_id_col,
        )
        self.figure_generator = FigureGenerator(
            self.data_processor,
            unique_row_id_col=self.unique_row_id_col,
        )
        self._selection_handler = PlotSelectionHandler(
            data_processor=self.data_processor,
            figure_generator=self.figure_generator,
            unique_row_id_col=self.unique_row_id_col,
            get_filtered_df=self._get_filtered_df,
            on_apply_selection=self._apply_selection_to_all_plots,
            on_update_label=self._set_selection_label_count,
        )

        self._table_container.clear()
        self._table_view = DataFrameTableView(
            new_df,
            unique_row_id_col=self.unique_row_id_col,
            on_row_selected=self._handle_table_row_selected,
        )
        with self._table_container:
            self._table_view.build()

        pre_filter_options = {
            col: [PRE_FILTER_NONE] + [str(v) for v in self.data_processor.get_pre_filter_values(col)]
            for col in self.pre_filter_columns
        }
        self._control_panel_container.clear()
        self._control_panel = PoolControlPanel(
            new_df,
            layout=self.layout,
            current_plot_index=self.current_plot_index,
            initial_state=self.plot_states[self.current_plot_index],
            on_any_change=self._on_any_change,
            on_layout_change=self._on_layout_change,
            on_save_config=self._save_config,
            on_plot_radio_change=self._on_plot_radio_change,
            on_apply_current_to_others=self._apply_current_to_others,
            on_replot_current=self._replot_current,
            on_reset_to_default=self._reset_to_default,
            on_x_column_selected=self._on_x_column_selected,
            on_y_column_selected=self._on_y_column_selected,
        )
        with self._control_panel_container:
            self._control_panel.build(pre_filter_options=pre_filter_options)

        self._rebuild_plot_panel()
        self._control_panel.sync_controls(
            self.plot_states[self.current_plot_index].plot_type,
            self.plot_states[self.current_plot_index].show_std_sem,
        )
        self._state_to_widgets(self.plot_states[self.current_plot_index])
        self._set_selection_label_count(len(self._selection_handler.get_selected_row_ids()))

    def _handle_table_row_selected(self, selected_row_id: str, row_dict: dict) -> None:
        """Handle row selection from DataFrameTableView.

        Args:
            selected_row_id: The unique row identifier (value from unique_row_id_col).
            row_dict: Dictionary containing all column values for the selected row.
        """
        # Call user-provided callback if set
        if self._on_table_row_selected:
            self._on_table_row_selected(selected_row_id, row_dict)
        
        # Clear existing plot selections and select the new point
        # This clears plot selections as requested
        self._selection_handler.handle_clear()
        
        # Select points in plots using the row_id
        
        self.select_points_by_row_id(selected_row_id)


    def _get_filtered_df(self, state: PlotState) -> pd.DataFrame:
        """Return dataframe filtered by state.pre_filter for selection/plot use."""
        return self.data_processor.filter_by_pre_filters(state.pre_filter)

    def _set_selection_label_count(self, count: int) -> None:
        """Update selection label text (callback from selection handler)."""
        if self._selection_label is not None:
            self._selection_label.text = f"{count} points selected" if count else "No selection"

    def _state_to_widgets(self, state: PlotState) -> None:
        """Populate all UI widgets from a PlotState."""
        if self._control_panel is None:
            return
        self._control_panel.bind_state(state)
        self._control_panel.sync_controls(state.plot_type, state.show_std_sem)

    def _widgets_to_state(self) -> PlotState:
        """Create PlotState from current UI widget values. xcol/ycol from current plot state (aggrid async)."""
        if self._control_panel is None:
            return self.plot_states[self.current_plot_index]
        current = self.plot_states[self.current_plot_index]
        return self._control_panel.get_state(xcol=current.xcol, ycol=current.ycol)

    # ----------------------------
    # UI
    # ----------------------------

    def build(self, *, container: Optional[ui.element] = None) -> None:
        """Build the main UI (public API). Call once to render the pool plot.

        Creates header, vertical splitter (table on top, controls+plots below),
        pre-filter dropdowns, control panel, and plot panel. Layout and plot state
        may be restored from saved config if present.

        Args:
            container: Optional NiceGUI container to build into. If None, widgets
                are created at the current top level.
        """
        # Helper to conditionally use container context or top-level
        def _build_content():
            # Header area at the top
            # with ui.column().classes("w-full"):
            #     with ui.row().classes("w-full items-center gap-3 flex-wrap"):
            #         ui.label("Pool Plot").classes("text-2xl font-bold mb-2")
            #         ui.button("Open CSV", on_click=self._on_open_csv).classes("text-sm")
            #         if self._on_refresh_requested is not None:
            #             ui.button("Refresh", on_click=self._on_refresh_click).classes("text-sm")

            # Vertical splitter: table view on top, plots/controls below
            # value=0 minimizes table (maximizes plots) on initial render
            self._verticalSplitter = ui.splitter(
                value=0,  # 0% for table (minimized), 100% for plots
                limits=(0, 100),  # Table can be 0-100% when user drags
                horizontal=True,  # Vertical split (horizontal=True means horizontal divider)
            ).classes("w-full h-screen")
            
            # TOP: Table view
            with self._verticalSplitter.before:
                self._table_container = ui.column().classes("w-full h-full min-h-0")
                with self._table_container:
                    self._table_view = DataFrameTableView(
                        self.df,
                        unique_row_id_col=self.unique_row_id_col,
                        on_row_selected=self._handle_table_row_selected,
                    )
                    self._table_view.build()

            # BOTTOM: Existing horizontal splitter with controls and plots
            with self._verticalSplitter.after:
                with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                    self._clicked_label = ui.label("Click a point to show the filtered df row...").classes("text-sm text-gray-600")
                    self._selection_label = ui.label("No selection").classes("text-sm font-medium")
                    ui.button("Clear selection", on_click=self._clear_selection).classes("text-sm")
                # Global Esc to clear selection (NiceGUI keyboard element)
                ui.keyboard(on_key=self._on_keyboard_key)
                # Main splitter: horizontal layout with controls on left, plot on right
                # on_change: when user resizes splitter, re-build plot panel so 1x2/2x1 layout is not lost
                # (NiceGUI/Quasar can re-render the 'after' slot and show only one plot; rebuild restores correct count)
                self._mainSplitter = ui.splitter(
                    value=self._control_panel_splitter_value,
                    limits=(0, 50),
                    on_change=lambda e: self._on_splitter_change(e),
                ).classes("w-full h-full")
            
            # LEFT: Control panel
            pre_filter_options = {
                col: [PRE_FILTER_NONE] + [str(v) for v in self.data_processor.get_pre_filter_values(col)]
                for col in self.pre_filter_columns
            }
            with self._mainSplitter.before:
                self._control_panel_container = ui.column().classes("w-full")
                with self._control_panel_container:
                    self._control_panel = PoolControlPanel(
                        self.df,
                        layout=self.layout,
                        current_plot_index=self.current_plot_index,
                        initial_state=self.plot_states[self.current_plot_index],
                        on_any_change=self._on_any_change,
                        on_layout_change=self._on_layout_change,
                        on_save_config=self._save_config,
                        on_plot_radio_change=self._on_plot_radio_change,
                        on_apply_current_to_others=self._apply_current_to_others,
                        on_replot_current=self._replot_current,
                        on_reset_to_default=self._reset_to_default,
                        on_x_column_selected=self._on_x_column_selected,
                        on_y_column_selected=self._on_y_column_selected,
                    )
                    self._control_panel.build(pre_filter_options=pre_filter_options)

            # RIGHT: Plot panel, then new vertical splitter (plot top, empty bottom) — same orientation as table|plots
            with self._mainSplitter.after:
                plot_empty_splitter_val = 75.0  # % for plot, rest empty
                with ui.splitter(
                    value=plot_empty_splitter_val,
                    limits=(20, 95),
                    horizontal=True,  # same as _verticalSplitter: top/bottom
                ).classes("w-full h-full") as plot_empty_splitter:
                    with plot_empty_splitter.before:
                        self._plot_container = ui.column().classes("w-full h-full")
                    with plot_empty_splitter.after:
                        ui.column().classes("w-full h-full min-h-0")  # empty pane below plot
            self._rebuild_plot_panel()

            self._control_panel.sync_controls(
                self.plot_states[self.current_plot_index].plot_type,
                self.plot_states[self.current_plot_index].show_std_sem,
            )
            self._state_to_widgets(self.plot_states[self.current_plot_index])
            self._set_selection_label_count(len(self._selection_handler.get_selected_row_ids()))
        
        # Build into container if provided, otherwise build at top level
        if container is not None:
            with container:
                _build_content()
        else:
            _build_content()

    def build_lazy(
        self,
        title: str,
        *,
        subtitle: Optional[str] = None,
        config: Optional[LazySectionConfig] = None,
    ) -> LazySection:
        """Build the UI inside a LazySection so it renders only when opened (public API).

        Same UI as build(), but wrapped in a ui.expansion. Content is created when
        the user opens the section, which avoids heavy work on initial page load.

        Args:
            title: Title shown on the expansion header.
            subtitle: Optional text shown below the title inside the section.
            config: Optional LazySectionConfig. If None, uses render_once=True,
                clear_on_close=False, show_spinner=True.

        Returns:
            LazySection instance. The section renders the plot controller UI on first open.

        Example:
            >>> ctrl = PlotPoolController(df, pre_filter_columns=["roi_id"])
            >>> section = ctrl.build_lazy("Pool Plot", subtitle="Click to load")
        """
        if config is None:
            config = LazySectionConfig(render_once=True, clear_on_close=False, show_spinner=True)
        
        return LazySection(
            title,
            subtitle=subtitle,
            render_fn=lambda container: self.build(container=container),
            config=config,
        )

    def _rebuild_plot_panel(self) -> None:
        """Rebuild the plot panel based on current layout.
        
        Clears existing plots and recreates them in the new layout.
        This is necessary because NiceGUI doesn't support dynamic
        restructuring of widget containers.
        """
        if self._plot_container is None:
            return
        
        # Clear all existing plot widgets from the container
        self._plot_container.clear()
        
        # Parse layout string (e.g., "1x2" -> rows=1, cols=2)
        rows, cols = map(int, self.layout.split('x'))
        
        # Initialize plot widgets list
        self._plots = []
        
        # Create layout based on rows/cols
        if rows == 1 and cols == 1:
            # 1x1: Single plot
            with self._plot_container:
                with ui.column().classes("w-full h-full min-h-0 p-4"):
                    plot = ui.plotly(
                        self._make_figure_dict(self.plot_states[0], selected_row_ids=None)
                    ).classes("w-full h-full")
                    plot.on("plotly_click", lambda e, idx=0: self._on_plotly_click(e, plot_index=idx))
                    if is_selection_compatible(self.plot_states[0].plot_type):
                        plot.on("plotly_relayout", lambda e, idx=0: self._on_plotly_relayout(e, plot_index=idx))
                    self._plots.append(plot)
        
        elif rows == 1 and cols == 2:
            # 1x2: Two plots side by side
            with self._plot_container:
                with ui.row().classes("w-full h-full gap-2 no-wrap"):
                    for i in range(2):
                        with ui.column().classes("flex-1 w-0 h-full min-h-0 p-4"):
                            plot = ui.plotly(
                                self._make_figure_dict(self.plot_states[i], selected_row_ids=None)
                            ).classes("w-full h-full")
                            plot.on("plotly_click", lambda e, idx=i: self._on_plotly_click(e, plot_index=idx))
                            if is_selection_compatible(self.plot_states[i].plot_type):
                                plot.on("plotly_relayout", lambda e, idx=i: self._on_plotly_relayout(e, plot_index=idx))
                            self._plots.append(plot)
        
        elif rows == 2 and cols == 1:
            # 2x1: Two plots stacked vertically
            with self._plot_container:
                with ui.column().classes("w-full h-full gap-2"):
                    for i in range(2):
                        with ui.column().classes("w-full flex-1 min-h-0 p-4"):
                            plot = ui.plotly(
                                self._make_figure_dict(self.plot_states[i], selected_row_ids=None)
                            ).classes("w-full h-full")
                            plot.on("plotly_click", lambda e, idx=i: self._on_plotly_click(e, plot_index=idx))
                            if is_selection_compatible(self.plot_states[i].plot_type):
                                plot.on("plotly_relayout", lambda e, idx=i: self._on_plotly_relayout(e, plot_index=idx))
                            self._plots.append(plot)
        
        elif rows == 2 and cols == 2:
            # 2x2: Four plots in a 2x2 grid
            with self._plot_container:
                with ui.column().classes("w-full h-full gap-2"):
                    for row_idx in range(2):
                        with ui.row().classes("w-full flex-1 gap-2 min-h-0 no-wrap"):
                            for col_idx in range(2):
                                plot_idx = row_idx * 2 + col_idx
                                with ui.column().classes("flex-1 w-0 h-full min-h-0 p-4"):
                                    plot = ui.plotly(
                                        self._make_figure_dict(self.plot_states[plot_idx], selected_row_ids=None)
                                    ).classes("w-full h-full")
                                    plot.on("plotly_click", lambda e, idx=plot_idx: self._on_plotly_click(e, plot_index=idx))
                                    if is_selection_compatible(self.plot_states[plot_idx].plot_type):
                                        plot.on("plotly_relayout", lambda e, idx=plot_idx: self._on_plotly_relayout(e, plot_index=idx))
                                    self._plots.append(plot)
        
        # Initialize last plot type tracking after plots are created
        if self._plots and self._last_plot_type is None:
            self._last_plot_type = self.plot_states[self.current_plot_index].plot_type

    # ----------------------------
    # Events
    # ----------------------------

    def _on_splitter_change(self, e=None) -> None:
        """Restore plot panel after splitter resize (avoids 1x2/2x1 collapsing to single plot). Persist splitter value."""
        self._rebuild_plot_panel()
        # Persist splitter value (throttled)
        if e is not None and hasattr(e, "value") and e.value is not None:
            try:
                self._control_panel_splitter_value = float(e.value)
                self._debounced_save_splitter()
            except (TypeError, ValueError):
                pass

    def _debounced_save_splitter(self) -> None:
        """Save splitter value to config after a short delay to avoid excessive writes."""
        if self._splitter_save_timer is not None:
            self._splitter_save_timer.cancel()
        self._splitter_save_timer = ui.timer(0.5, self._flush_save_splitter, once=True)

    def _flush_save_splitter(self) -> None:
        """Actually persist the splitter value to config."""
        self._splitter_save_timer = None
        try:
            from nicewidgets.plot_pool_widget.pool_plot_config import PoolPlotConfig
            config = PoolPlotConfig.load()
            config.set_control_panel_splitter_value(self._control_panel_splitter_value)
            config.save()
        except Exception as ex:
            logger.warning(f"Failed to save splitter config: {ex}")

    def _get_num_plots_for_layout(self, layout_str: str) -> int:
        """Get number of plots needed for a layout string.
        
        Args:
            layout_str: Layout string like "1x1", "1x2", "2x1", "2x2".
            
        Returns:
            Number of plots needed.
        """
        rows, cols = map(int, layout_str.split('x'))
        return rows * cols
    
    def _on_plot_radio_change(self, e) -> None:
        """Handle plot radio button change with validation.
        
        Validates that the selected plot index is valid for the current layout.
        If invalid, reverts to a valid selection.
        """
        try:
            selected_plot_num = int(e.value)
            plot_index = selected_plot_num - 1
            
            # Validate that this plot index is available for current layout
            num_plots = self._get_num_plots_for_layout(self.layout)
            if plot_index < 0 or plot_index >= num_plots:
                logger.warning(f"Invalid plot selection {selected_plot_num} for layout {self.layout} (max {num_plots} plots)")
                if self._control_panel and self._control_panel._plot_radio:
                    self._control_panel._plot_radio.value = str(self.current_plot_index + 1)
                ui.notify(f"Plot {selected_plot_num} is not available for layout {self.layout}", type="warning")
                return
            self._on_plot_selection_change(plot_index)
        except (ValueError, TypeError) as ex:
            logger.error(f"Error parsing plot radio selection: {ex}")
            if self._control_panel and self._control_panel._plot_radio:
                self._control_panel._plot_radio.value = str(self.current_plot_index + 1)
    
    def _on_layout_change(self, layout_str: str) -> None:
        """Handle layout change (1x1, 1x2, 2x1, 2x2).
        
        Args:
            layout_str: Layout string like "1x1", "1x2", "2x1", "2x2".
        """
        logger.info(f"Layout changed to: {layout_str}")
        # Save current plot state before changing layout
        self.plot_states[self.current_plot_index] = self._widgets_to_state()
        
        # Ensure we have enough plot states for the new layout
        num_plots_needed = self._get_num_plots_for_layout(layout_str)
        current_state = self.plot_states[self.current_plot_index]
        
        # Pad plot_states if needed
        while len(self.plot_states) < num_plots_needed:
            self.plot_states.append(PlotState.from_dict(current_state.to_dict()))
        
        # Clamp current_plot_index if needed
        if self.current_plot_index >= num_plots_needed:
            self.current_plot_index = num_plots_needed - 1
        
        self.layout = layout_str
        
        # Ensure current selection is valid
        num_plots = self._get_num_plots_for_layout(layout_str)
        if self.current_plot_index >= num_plots:
            self.current_plot_index = num_plots - 1
        
        if self._control_panel and self._control_panel._plot_radio:
            self._control_panel._plot_radio.value = str(self.current_plot_index + 1)
        self._rebuild_plot_panel()
    
    def _on_plot_selection_change(self, plot_index: int) -> None:
        """Handle plot selection change (switch which plot is being edited).
        
        Args:
            plot_index: Index of plot to switch to (0-based).
        """
        logger.info(f"Switching to plot {plot_index + 1}")
        # Save current plot state
        self.plot_states[self.current_plot_index] = self._widgets_to_state()
        # Update current index
        self.current_plot_index = plot_index
        # Load new plot state into widgets (no replot needed)
        self._state_to_widgets(self.plot_states[self.current_plot_index])
    
    def _apply_current_to_others(self) -> None:
        """Apply current plot's state to all other plots."""
        logger.info("Applying current plot state to all other plots")
        # Get current state from widgets
        current_state = self._widgets_to_state()
        self.plot_states[self.current_plot_index] = current_state
        
        # Copy to all other plots
        for i in range(len(self.plot_states)):
            if i != self.current_plot_index:
                # Deep copy using serialization
                self.plot_states[i] = PlotState.from_dict(current_state.to_dict())
        
        # Replot all visible plots
        self._replot_all()

    def _save_config(self) -> None:
        """Save current layout and all plot states to config file."""
        from nicewidgets.plot_pool_widget.pool_plot_config import PoolPlotConfig
        
        # Save current plot state before saving
        self.plot_states[self.current_plot_index] = self._widgets_to_state()
        
        # Load existing config and update layout and plot_states
        config = PoolPlotConfig.load()
        config.set_layout(self.layout)
        config.set_plot_states(self.plot_states)
        config.save()
        
        ui.notify(f"Plot configuration saved (layout: {self.layout}, {len(self.plot_states)} plot(s))", type="positive")

    def _reset_to_default(self) -> None:
        """Reset all plots to the default plot state."""
        logger.info("Resetting all plots to default state")
        # Apply default state to all plots
        for i in range(len(self.plot_states)):
            self.plot_states[i] = PlotState.from_dict(self.default_plot_state.to_dict())
        
        # Update widgets to reflect default state
        self._state_to_widgets(self.plot_states[self.current_plot_index])
        
        # Replot all visible plots
        self._replot_all()
        ui.notify("Plots reset to default configuration", type="info")

    def _on_x_column_selected(self, row_dict: dict[str, Any]) -> None:
        """Callback when X column is selected in aggrid."""
        column_name = row_dict.get("column")
        if not column_name:
            logger.warning(f"X column selection callback received invalid row_dict: {row_dict}")
            return
        column_name = str(column_name)
        # Box/Violin/Swarm don't use xcol for x-axis (they use group_col), so no validation needed here
        # X column selection is still allowed but won't affect these plot types
        logger.info(f"X column selected: {column_name}")
        self.plot_states[self.current_plot_index].xcol = column_name
        self._on_any_change()

    def _on_y_column_selected(self, row_dict: dict[str, Any]) -> None:
        """Callback when Y column is selected in aggrid."""
        column_name = row_dict.get("column")
        if column_name:
            logger.info(f"Y column selected: {column_name}")
            self.plot_states[self.current_plot_index].ycol = str(column_name)
            self._on_any_change()
        else:
            logger.warning(f"Y column selection callback received invalid row_dict: {row_dict}")

    def _on_any_change(self, _e=None) -> None:
        """Handle changes to any control widget and trigger replot."""
        if self._control_panel is None:
            return
        logger.info("Control change detected, updating state and replotting")
        new_state = self._widgets_to_state()
        new_state.xcol = self.plot_states[self.current_plot_index].xcol
        new_state.ycol = self.plot_states[self.current_plot_index].ycol

        if new_state.plot_type in (PlotType.BOX_PLOT, PlotType.VIOLIN, PlotType.SWARM):
            if not new_state.group_col or new_state.group_col == "(none)":
                ui.notify(
                    "Box/Violin/Swarm plots require a Group/Color column for x-axis. Select a categorical column.",
                    type="warning",
                )
                new_state.group_col = self.plot_states[self.current_plot_index].group_col
                self._control_panel._group_select.value = new_state.group_col if new_state.group_col else "(none)"
            elif not is_categorical_column(self.df, new_state.group_col):
                ui.notify(
                    "Group/Color column must be categorical for Box/Violin/Swarm plots.",
                    type="warning",
                )
                new_state.group_col = self.plot_states[self.current_plot_index].group_col
                self._control_panel._group_select.value = new_state.group_col if new_state.group_col else "(none)"
            if new_state.color_grouping and not is_categorical_column(self.df, new_state.color_grouping):
                ui.notify("Group/Nesting column must be categorical.", type="warning")
                new_state.color_grouping = self.plot_states[self.current_plot_index].color_grouping
                self._control_panel._color_grouping_select.value = new_state.color_grouping if new_state.color_grouping else "(none)"

        self.plot_states[self.current_plot_index] = new_state
        self._control_panel.sync_controls(new_state.plot_type, new_state.show_std_sem)
        self._replot_current()

    def _clear_selection(self) -> None:
        """Clear linked selection and refresh all compatible plots."""
        self._selection_handler.handle_clear()

    def _on_open_csv(self) -> None:
        """Open a CSV file via native dialog. Replace with pywebview callback when available."""
        # Dummy for now: wire pywebview native file-open callback here
        ui.notify("Open CSV: connect pywebview file dialog callback here", type="info")

    def _on_refresh_click(self) -> None:
        """Handle Refresh button click: call on_refresh_requested callback and update the pool."""
        if self._on_refresh_requested is None:
            return
        try:
            new_df = self._on_refresh_requested()
            self.update_df(new_df)
            ui.notify("Data refreshed", type="positive")
        except Exception as e:
            logger.exception("Refresh failed")
            ui.notify(f"Refresh failed: {e}", type="negative")

    def _on_keyboard_key(self, e) -> None:
        key_name = getattr(getattr(e, "key", None), "name", None) if e else None
        action = getattr(e, "action", None)
        self._selection_handler.handle_key(key_name, action)

    def _on_plotly_relayout(self, e: GenericEventArguments, plot_index: int = 0) -> None:
        raw = e.args
        if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
            payload = raw[0]
        elif isinstance(raw, dict):
            payload = raw
        else:
            payload = {}
        state = self.plot_states[plot_index]
        self._selection_handler.handle_relayout(payload, plot_index, state)

    def _apply_selection_to_all_plots(self) -> None:
        """Update all selection-compatible plots to show the current linked selection."""
        selected = self._selection_handler.get_selected_row_ids()
        for i in range(len(self._plots)):
            if i >= len(self.plot_states):
                break
            if not is_selection_compatible(self.plot_states[i].plot_type):
                continue
            fig_dict = self._make_figure_dict(self.plot_states[i], selected_row_ids=selected)
            self._plots[i].update_figure(fig_dict)
            self._plots[i].update()

    def _on_plotly_click(self, e: GenericEventArguments, plot_index: int = 0) -> None:
        """Handle click events on the Plotly plot.
        
        Args:
            e: Event arguments from Plotly click.
            plot_index: Index of the plot that was clicked (0-based).
        """
        points = (e.args or {}).get("points") or []
        if not points:
            logger.warning("Plotly click event received but no points found")
            return

        p0: dict[str, Any] = points[0]
        custom = p0.get("customdata")
        state = self.plot_states[plot_index]

        # per-row plots: click -> row_id -> filtered df row
        if state.plot_type in {PlotType.SCATTER, PlotType.SWARM}:
            row_id: Optional[str] = None
            if isinstance(custom, (str, int, float)):
                row_id = str(custom)
            elif isinstance(custom, (list, tuple)) and custom:
                row_id = str(custom[0])

            if not row_id:
                logger.warning(f"Could not extract row_id from plotly click: custom={custom}")
                return

            # logger.info(f"Plotly click on plot {plot_index + 1}: plot_type={state.plot_type.value}, row_id={row_id}")

            df_f = self._get_filtered_df(state)
            idx = self._id_to_index_filtered.get(row_id)
            if idx is None:
                logger.warning(f"Row ID {row_id} not found in filtered index")
                return

            row = df_f.iloc[idx]
            if self._clicked_label:
                self._clicked_label.text = f"Plot {plot_index + 1}: {format_pre_filter_display(state.pre_filter)} \n {self.unique_row_id_col}={row_id} \n (filtered iloc={idx})"
            row_dict = row.to_dict()
            if self._on_table_row_selected:
                self._on_table_row_selected(row_id, row_dict)
            return

        # grouped aggregation plot: click -> group summary
        if state.plot_type == PlotType.GROUPED:
            x = p0.get("x")
            y = p0.get("y")
            logger.info(f"Plotly click on plot {plot_index + 1} grouped plot: group={x}, y={y}")
            if self._clicked_label:
                self._clicked_label.text = f"Plot {plot_index + 1}: {format_pre_filter_display(state.pre_filter)} clicked group={x}, y={y} (aggregated)"
            return

    # ----------------------------
    # Plotting
    # ----------------------------

    def _replot_current(self) -> None:
        """Update the current plot with its state and data."""
        if not self._plots:
            logger.warning("Cannot replot: no plots available")
            return
        
        # Clamp current_plot_index to valid range based on actual number of plots
        if self.current_plot_index >= len(self._plots):
            logger.debug(f"Clamping current_plot_index from {self.current_plot_index} to {len(self._plots) - 1}")
            self.current_plot_index = len(self._plots) - 1
        
        state = self.plot_states[self.current_plot_index]
        
        # Check if plot type changed - if so, force a full rebuild
        plot_type_changed = (
            self._last_plot_type is not None and 
            self._last_plot_type != state.plot_type
        )
        
        logger.info(
            f"Replotting plot {self.current_plot_index + 1} - "
            f"plot_type={state.plot_type.value}, pre_filter={state.pre_filter}, "
            f"xcol={state.xcol}, ycol={state.ycol}, group_col={state.group_col}, "
            f"show_raw={state.show_raw}, show_legend={state.show_legend}, "
            f"plot_type_changed={plot_type_changed}"
        )
        
        try:
            figure_dict = self._make_figure_dict(state)
            logger.info(f"Figure dict created successfully with {len(figure_dict.get('data', []))} traces")
            
            # If plot type changed, force a full rebuild of the plot panel
            # This is necessary because update_figure() can be unreliable when plot structure changes significantly
            if plot_type_changed:
                logger.info(f"Plot type changed from {self._last_plot_type.value} to {state.plot_type.value}, forcing full rebuild")
                self._rebuild_plot_panel()
            else:
                # Normal update - just update the figure
                self._plots[self.current_plot_index].update_figure(figure_dict)
                # Explicitly call update() to ensure the change is applied
                self._plots[self.current_plot_index].update()
            
            # Update last plot type
            self._last_plot_type = state.plot_type
            
            logger.info(f"Plot {self.current_plot_index + 1} updated successfully")
        except Exception as ex:
            logger.exception(f"Error replotting plot {self.current_plot_index + 1}: {ex}")
    
    def _replot_all(self) -> None:
        """Replot all visible plots based on current layout."""
        rows, cols = map(int, self.layout.split('x'))
        num_plots = rows * cols
        
        selected = self._selection_handler.get_selected_row_ids()
        for i in range(min(num_plots, len(self._plots), len(self.plot_states))):
            state = self.plot_states[i]
            self._plots[i].update_figure(
                self._make_figure_dict(state, selected_row_ids=selected or None)
            )

    def _make_figure_dict(
        self,
        state: PlotState,
        *,
        selected_row_ids: Optional[set[str]] = None,
    ) -> dict:
        """Generate Plotly figure dictionary based on plot state.
        
        Args:
            state: PlotState to use for generating the figure.
            selected_row_ids: If set, these row_ids are shown as selected (linked selection).
            
        Returns:
            Plotly figure dictionary.
        """
        df_f = self._get_filtered_df(state)
        self._id_to_index_filtered = self.data_processor.build_row_id_index(df_f)
        
        logger.debug(f"Making figure: plot_type={state.plot_type.value}, filtered_rows={len(df_f)}")
        figure_dict = self.figure_generator.make_figure(
            df_f, state, selected_row_ids=selected_row_ids or self._selection_handler.get_selected_row_ids() or None
        )
        logger.debug(f"Figure generated: {len(figure_dict.get('data', []))} traces")
        return figure_dict
