"""Demo controller wiring NicePool to synthetic velocity-pool data.

This module owns demo state (current dataset, last selected row) and wires
NicePool callbacks to demo-only UI. It follows the nicewidgets host-application
pattern: the host builds the widget, passes callbacks, and drives the widget
through its public API (``set_dataframe``, ``select_points_by_row_ids``,
``set_dark_mode``).

Only nicewidgets and NiceGUI are imported, so the controller reads as a
template for real host applications.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from nicewidgets.nicepool import NicePool, NicePoolConfig
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE
from nicewidgets.utils.logging import get_logger

try:
    from examples.nicepool.sample_data import (
        PRE_FILTER_COLUMNS,
        UNIQUE_ROW_ID_COL,
        SampleDataCatalog,
    )
except ImportError:
    from sample_data import (  # type: ignore[no-redef]
        PRE_FILTER_COLUMNS,
        UNIQUE_ROW_ID_COL,
        SampleDataCatalog,
    )

logger = get_logger(__name__)

# First-run plot: a swarm of velocity_mean grouped by experiment (grandparent).
# Same shape as a saved session/preset payload, so host apps get a useful first
# paint without depending on user disk state.
INITIAL_PLOT_CONFIG: dict[str, Any] = {
    "layout": "1x1",
    "plot_states": [
        {
            "pre_filter": {
                "accept": PRE_FILTER_NONE,
                "channel": PRE_FILTER_NONE,
                "roi_id": PRE_FILTER_NONE,
            },
            "xcol": "grandparent",
            "ycol": "velocity_mean",
            "plot_type": "swarm",
            "group_col": "grandparent",
            "color_grouping": None,
            "show_mean": True,
            "show_std_sem": True,
        }
    ],
}


class NicePoolDemoController:
    """Own demo state and wire NicePool callbacks to demo-only controls.

    Args:
        catalog: Sample data source providing named velocity-pool DataFrames.
    """

    def __init__(self, catalog: SampleDataCatalog, *, dark_mode: bool = False) -> None:
        self._catalog = catalog
        self._dataset_name: str = catalog.names[0]
        self._dark_mode: bool = bool(dark_mode)
        self._pool: NicePool | None = None

        self._selection_label: ui.label | None = None

    @property
    def dataset_name(self) -> str:
        """Return the currently loaded dataset name."""
        return self._dataset_name

    @property
    def pool(self) -> NicePool:
        """Return the built NicePool widget.

        Raises:
            RuntimeError: If accessed before :meth:`build`.
        """
        if self._pool is None:
            raise RuntimeError("NicePool has not been built yet; call build() first.")
        return self._pool

    def build(self) -> None:
        """Build the NicePool widget for the current dataset."""
        df = self._catalog.get_dataframe(self._dataset_name)
        self._pool = NicePool(
            df,
            config=NicePoolConfig(
                unique_row_id_col=UNIQUE_ROW_ID_COL,
                pre_filter_columns=list(PRE_FILTER_COLUMNS),
                initial_plot_config=INITIAL_PLOT_CONFIG,
                show_table_widget=True,
                show_selection_feedback=True,
                show_save_button=False,
                enable_config_persistence=False,
                enable_plot_presets=True,
                dark_mode=self._dark_mode,
            ),
            on_row_selected=self._on_row_selected,
            on_refresh_requested=self._on_refresh_requested,
        )
        self._pool.build()

    def bind_selection_label(self, label: ui.label) -> None:
        """Bind a demo label used to echo the last selected row."""
        self._selection_label = label

    def load_dataset(self, name: str) -> None:
        """Swap the widget's DataFrame to another named dataset.

        Args:
            name: Dataset name from the catalog.
        """
        self._dataset_name = name
        self.pool.set_dataframe(self._catalog.get_dataframe(name))
        self._set_selection_text("(none)")

    def select_accepted_rows(self) -> None:
        """Programmatically highlight every accepted row in the plots."""
        df = self._catalog.get_dataframe(self._dataset_name)
        row_ids = df.loc[df["accept"] == True, UNIQUE_ROW_ID_COL].tolist()  # noqa: E712
        self.pool.select_points_by_row_ids(row_ids)

    def set_dark_mode(self, enabled: bool) -> None:
        """Toggle the Plotly layout theme.

        Args:
            enabled: Whether dark mode is enabled.
        """
        self._dark_mode = enabled
        if self._pool is not None:
            self._pool.set_dark_mode(enabled)

    def _on_row_selected(self, row_id: str, row: dict[str, object]) -> None:
        """Echo the selected row id and a couple of fields to the demo label."""
        velocity = row.get("velocity_mean")
        velocity_text = f"{velocity:.1f}" if isinstance(velocity, (int, float)) else "n/a"
        self._set_selection_text(
            f"{row_id}  (grandparent={row.get('grandparent')!r}, "
            f"velocity_mean={velocity_text})"
        )

    def _on_refresh_requested(self):
        """Return a fresh DataFrame for the current dataset (refresh button)."""
        self._set_selection_text("(none)")
        return self._catalog.get_dataframe(self._dataset_name)

    def _set_selection_text(self, text: str) -> None:
        if self._selection_label is not None:
            self._selection_label.text = f"Selected: {text}"
