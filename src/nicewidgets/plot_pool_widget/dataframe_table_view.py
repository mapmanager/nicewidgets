"""DataFrame table view widget for displaying and selecting rows from a dataframe.

This module provides DataFrameTableView class for displaying a pandas DataFrame
in a NiceGUI aggrid widget with row selection capabilities.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd
from nicegui import ui

from nicewidgets.plot_pool_widget.lazy_section import LazySection, LazySectionConfig
from nicewidgets.plot_pool_widget.plot_helpers import _ensure_aggrid_compact_css
from nicewidgets.utils.logging import get_logger
from nicewidgets.gold_standard_aggrid import gold_standard_aggrid

logger = get_logger(__name__)


class DataFrameTableView:
    """Table view widget for displaying and selecting rows from a DataFrame.
    
    Displays a pandas DataFrame in a NiceGUI aggrid with single row selection.
    When a row is selected, extracts the unique_row_id_col value and calls the
    on_row_selected callback with (unique_row_id, row_dict).
    
    Attributes:
        df: The source DataFrame.
        unique_row_id_col: Column name containing unique row identifiers.
        on_row_selected: Optional callback called when a row is selected.
            Signature: (unique_row_id: str, row_dict: dict) -> None
        _aggrid: The underlying ui.aggrid widget (set after build()).
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        unique_row_id_col: str = "path",
        on_row_selected: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> None:
        """Initialize DataFrameTableView with dataframe and configuration.
        
        Args:
            df: DataFrame to display in the table.
            unique_row_id_col: Column name containing unique row identifiers.
            on_row_selected: Optional callback called when a row is selected.
                Receives (unique_row_id: str, row_dict: dict) where unique_row_id is the
                value from unique_row_id_col and row_dict is the full row data.
        """
        self.df = df
        self.unique_row_id_col = unique_row_id_col
        self._on_row_selected = on_row_selected
        self._aggrid: Optional[ui.aggrid] = None
        
        # Validate unique_row_id_col exists
        if self.unique_row_id_col not in df.columns:
            raise ValueError(f"unique_row_id_col '{unique_row_id_col}' not found in dataframe columns")
    
    def build(self, *, container: Optional[ui.element] = None) -> None:
        """Build the aggrid table UI using ui.aggrid.from_pandas(df).
        
        Args:
            container: Optional container element to build into. If provided,
                the aggrid will be created inside this container. If None,
                widgets are created at the top level.
        """
        _ensure_aggrid_compact_css()

        def _build_aggrid():
            self._aggrid = gold_standard_aggrid(self.df,unique_row_id_col=self.unique_row_id_col, row_select_callback=self._on_row_selected)
            return

        if container is not None:
            with container:
                _build_aggrid()
        else:
            _build_aggrid()
    
    def build_lazy(
        self,
        title: str,
        *,
        subtitle: Optional[str] = None,
        config: Optional[LazySectionConfig] = None,
    ) -> LazySection:
        """Build table wrapped in a LazySection for lazy rendering.
        
        This is a convenience method that wraps build() in a LazySection.
        The table will only be rendered when the expansion is opened.
        
        Args:
            title: Title for the expansion section.
            subtitle: Optional subtitle text shown below the title.
            config: Optional LazySectionConfig. If None, uses defaults
                (render_once=True, clear_on_close=False, show_spinner=True).
        
        Returns:
            LazySection instance containing the table view.
        """
        if config is None:
            config = LazySectionConfig(render_once=True, clear_on_close=False, show_spinner=True)
        
        return LazySection(
            title,
            subtitle=subtitle,
            render_fn=lambda container: self.build(container=container),
            config=config,
        )
    
    def get_selected_row_id(self) -> Optional[str]:
        """Get currently selected row_id (if any).
        
        Returns:
            The row_id value from the selected row, or None if no row is selected.
        """
        if self._aggrid is None:
            return None
        try:
            # Note: This is async but we can't await here, so return None for now
            # In practice, selection is handled via the callback
            return None
        except Exception:
            return None
