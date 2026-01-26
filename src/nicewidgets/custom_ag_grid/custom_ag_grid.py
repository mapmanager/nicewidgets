# grid_gpt.py

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Iterable, List, Mapping, Union, TYPE_CHECKING

from nicegui import events, ui

from nicewidgets.utils.logging import get_logger
from nicewidgets.custom_ag_grid.config import ColumnConfig, GridConfig, SelectionMode
from nicewidgets.custom_ag_grid.theme import ensure_aggrid_theme

logger = get_logger(__name__)

# Optional pandas
try:  # pragma: no cover - import guard
    import pandas as _pd  # type: ignore[import]
    HAS_PANDAS = True
except Exception:  # pragma: no cover - pandas optional
    _pd = None  # type: ignore[assignment]
    HAS_PANDAS = False

# Optional polars
try:  # pragma: no cover - import guard
    import polars as _pl  # type: ignore[import]
    HAS_POLARS = True
except Exception:  # pragma: no cover - polars optional
    _pl = None  # type: ignore[assignment]
    HAS_POLARS = False

if TYPE_CHECKING:  # for type checkers only
    import pandas as pd
    import polars as pl
else:  # runtime aliases (may be None)
    pd = _pd  # type: ignore[assignment]
    pl = _pl  # type: ignore[assignment]


RowDict = dict[str, Any]
RowsLike = List[RowDict]
DataLike = Union[RowsLike, "pd.DataFrame", "pl.DataFrame"]  # type: ignore[name-defined]

CellEditedHandler = Callable[[int, str, Any, Any, dict[str, Any]], None]
RowSelectedHandler = Callable[[int, dict[str, Any]], None]


class CustomAgGrid:
    """NiceGUI AG Grid wrapper backed by a list of row dictionaries.

    This widget accepts data from multiple sources (list of dictionaries,
    pandas DataFrame, or Polars DataFrame) and exposes a configurable
    AG Grid instance with editable cells and row selection.

    Events (via callback registration):
        on_cell_edited(handler):
            Handler called as
            ``handler(row_index, field_name, old_value, new_value, full_row_dict)``.
        on_row_selected(handler):
            Handler called as
            ``handler(row_index, full_row_dict)``.
    """

    def __init__(
        self,
        data: DataLike,
        columns: list[ColumnConfig],
        grid_config: GridConfig | None = None,
        parent: ui.element | None = None,
    ) -> None:
        """Initialize a new CustomAgGrid widget.

        Args:
            data: Input data. May be a list of row dictionaries, a pandas
                DataFrame, or a Polars DataFrame. Internally converted to
                a list-of-dicts representation.
            columns: Declarative column configuration for the grid.
            grid_config: Grid-level configuration such as selection mode,
                height, row height, and theme. If ``None``, a default
                GridConfig is used.
            parent: Optional parent NiceGUI element. If provided, the grid
                is created inside this container. If ``None``, a new
                ``ui.column`` container is created.
        """
        ensure_aggrid_theme()

        self._rows: RowsLike = self._convert_input_to_rows(data)
        self._columns: list[ColumnConfig] = columns
        self._grid_config: GridConfig = grid_config or GridConfig()

        # Track last known selected rows (simple, works well for single selection).
        self._last_selected_rows: RowsLike = []
        # stores last row dict(s) we consider “selected”

        self._selection_origin = "internal"
        # prevents feedback loops from programmatic selection
        # 
        self._last_row_id = None
        # dedupe repeated "rowSelected" events for same row
        
        # Local callback registries – these live and die with this widget instance.
        self._cell_edited_handlers: list[CellEditedHandler] = []
        self._row_selected_handlers: list[RowSelectedHandler] = []

        # Container with theme and style toggles
        container_classes = (
            f"w-full {self._grid_config.height} {self._grid_config.theme_class}"
        )
        if self._grid_config.zebra_rows:
            container_classes += " aggrid-zebra"
        if self._grid_config.hover_highlight:
            container_classes += " aggrid-hover"
        else:
            container_classes += " aggrid-no-hover"
        if self._grid_config.tight_layout:
            container_classes += " aggrid-tight"

        self._container: ui.element = parent or ui.column()
        self._container.classes(container_classes)

        column_defs = self._build_column_defs()
        grid_options = self._build_grid_options(column_defs, self._rows)

        # import pprint as pprint
        # print('grid options:')
        # pprint.pprint(grid_options)

        self._grid: ui.aggrid = ui.aggrid(grid_options)
        self._grid.on("cellValueChanged", self._on_cell_value_changed)
        # self._grid.on("rowSelected", self._on_row_selected)  # was this
        self._grid.on("cellClicked", self._on_row_selected)
        self._grid.on("cellKeyDown", self._on_cell_key_down)


        logger.info(
            f"CustomAgGrid initialized: {len(self._rows)} rows, "
            f"{len(self._columns)} columns, "
            f"selection_mode={self._grid_config.selection_mode}"
        )

    # ------------------------------------------------------------------
    # Public properties and accessors
    # ------------------------------------------------------------------

    @property
    def grid(self) -> ui.aggrid:
        """Return the underlying NiceGUI ``ui.aggrid`` instance.

        This can be useful when advanced AG Grid options or event wiring
        are required beyond what the wrapper currently exposes.
        """
        return self._grid

    @property
    def rows(self) -> RowsLike:
        """Return a shallow copy of the current row data.

        Returns:
            A list of dictionaries representing the current table contents.
            Modifying the returned list or its dictionaries will not
            automatically update the grid.
        """
        return [row.copy() for row in self._rows]

    def to_rows(self) -> RowsLike:
        """Alias for :attr:`rows` to make the API more explicit."""
        return self.rows

    def to_pandas(self) -> "pd.DataFrame":
        """Return the current table contents as a pandas DataFrame.

        Returns:
            A new pandas DataFrame constructed from the internal row list.
            Changes to the returned DataFrame will not affect the grid.

        Raises:
            ImportError: If pandas is not installed in the environment.
        """
        if not HAS_PANDAS:
            raise ImportError(
                "pandas is not available. Install 'pandas' to use to_pandas()."
            )
        assert pd is not None  # for type checkers
        return pd.DataFrame(self._rows)

    def to_polars(self) -> "pl.DataFrame":
        """Return the current table contents as a Polars DataFrame.

        Returns:
            A new Polars DataFrame constructed from the internal row list.

        Raises:
            ImportError: If Polars is not installed in the environment.
        """
        if not HAS_POLARS:
            raise ImportError(
                "Polars is not available. Install 'polars' to use to_polars()."
            )
        assert pl is not None  # for type checkers
        return pl.DataFrame(self._rows)

    # ------------------------------------------------------------------
    # Public event registration API
    # ------------------------------------------------------------------

    def on_cell_edited(self, handler: CellEditedHandler) -> None:
        """Register a callback for cell edit events.

        The handler is called with:
            (row_index, field_name, old_value, new_value, full_row_dict)
        """
        self._cell_edited_handlers.append(handler)

    def on_row_selected(self, handler: RowSelectedHandler) -> None:
        """Register a callback for row selection events.

        The handler is called with:
            (row_index, full_row_dict)
        """
        self._row_selected_handlers.append(handler)

    # ------------------------------------------------------------------
    # Data replacement / refresh
    # ------------------------------------------------------------------

    def set_data(self, data: DataLike) -> None:
        """Replace the grid data and refresh the view.

        Args:
            data: New data to display. May be a list of dictionaries,
                a pandas DataFrame, or a Polars DataFrame.
        """
        self._rows = self._convert_input_to_rows(data)
        self._grid.options["rowData"] = self._rows
        self._grid.update()
        # logger.debug(f"set_data: updated grid with {len(self._rows)} rows")

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def get_selected_rows(self) -> RowsLike:
        """Return the last known selected rows.

        Returns:
            A list of dictionaries representing the most recently selected
            rows. For ``selection_mode="single"``, this list will contain
            at most one row.
        """
        return [row.copy() for row in self._last_selected_rows]

    # ------------------------------------------------------------------
    # Internal: input conversion
    # ------------------------------------------------------------------

    def _convert_input_to_rows(self, data: DataLike) -> RowsLike:
        """Convert input data into the canonical ``list[dict]`` representation.

        Args:
            data: Input data as list of dicts, pandas DataFrame, or
                Polars DataFrame.

        Returns:
            A list of dictionaries suitable for AG Grid's ``rowData``.
        """
        # list-of-dicts (or similar sequence of mappings)
        if isinstance(data, list):
            if all(isinstance(row, Mapping) for row in data):
                return [dict(row) for row in data]  # ensure plain dicts
            raise TypeError("List input must contain mapping/dict-like rows.")

        # pandas DataFrame
        if HAS_PANDAS and pd is not None and isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")

        # Polars DataFrame
        if HAS_POLARS and pl is not None and isinstance(data, pl.DataFrame):
            return data.to_dicts()

        raise TypeError(
            "Unsupported data type for CustomAgGrid. "
            "Expected list[dict], pandas.DataFrame, or polars.DataFrame."
        )

    # ------------------------------------------------------------------
    # Internal: AG Grid option building
    # ------------------------------------------------------------------

    def _build_column_defs(self) -> list[dict[str, Any]]:
        """Convert ColumnConfig instances into AG Grid ``columnDefs``.

        Returns:
            A list of dictionaries that can be passed as ``columnDefs`` to
            AG Grid.
        """
        defs: list[dict[str, Any]] = []

        if self._grid_config.show_row_index:
            defs.append(
                {
                    "headerName": self._grid_config.row_index_header,
                    "valueGetter": "node.rowIndex + 1",
                    "editable": False,
                    "sortable": False,
                    "filter": False,
                    "resizable": False,
                    "width": self._grid_config.row_index_width,
                    "pinned": "left",
                    "cellClass": "ag-cell-right",
                }
            )

        for col in self._columns:
            col_def: dict[str, Any] = {
                "headerName": col.header or col.field,
                "field": col.field,
                "editable": col.editable,
                "sortable": col.sortable,
                "filter": col.filterable,
                "resizable": col.resizable,
            }

            # Editor selection
            if col.editor == "select" and col.editable:
                values: list[str]
                if col.choices == "unique":
                    # Collect unique values from current data for this column.
                    seen: set[str] = set()
                    for row in self._rows:
                        if col.field in row and row[col.field] is not None:
                            seen.add(str(row[col.field]))
                    values = sorted(seen)
                elif col.choices is not None:
                    values = [str(v) for v in col.choices]
                else:
                    values = []

                # NiceGUI: colon prefix means "treat value as a JS expression".
                col_def[":cellEditor"] = "'agSelectCellEditor'"
                col_def["cellEditorParams"] = {"values": values}

            # Let callers inject arbitrary AG Grid options, if needed.
            col_def.update(col.extra_grid_options)
            defs.append(col_def)

        return defs

    def _build_grid_options(
        self,
        column_defs: list[dict[str, Any]],
        row_data: RowsLike,
    ) -> dict[str, Any]:
        """Build the AG Grid options dictionary.

        Args:
            column_defs: Column definition dictionaries for AG Grid.
            row_data: List of row dictionaries for AG Grid.

        Returns:
            A dictionary suitable for passing to ``ui.aggrid``.
        """
        selection: SelectionMode = self._grid_config.selection_mode
        if selection == "none":
            row_selection: str | None = None
        elif selection == "single":
            row_selection = "single"
        else:
            row_selection = "multiple"

        opts: dict[str, Any] = {
            "columnDefs": column_defs,
            "rowData": row_data,
            "defaultColDef": {
                "sortable": True,
                "filter": True,
                "resizable": True,
            },
            "rowHeight": self._grid_config.row_height,
            "headerHeight": self._grid_config.header_height,
        }

        if row_selection is not None:
            opts["rowSelection"] = row_selection

        if self._grid_config.stop_editing_on_focus_loss:
            opts["stopEditingWhenCellsLoseFocus"] = True

        if not self._grid_config.hover_highlight:
            opts["suppressRowHoverHighlight"] = True

        # Stable row id mapping for run_row_method + selection persistence
        if self._grid_config.row_id_field:
            field = self._grid_config.row_id_field
            opts[":getRowId"] = f"(params) => params.data && params.data['{field}']"
            # logger.warning(f'opts[":getRowId"] is: "{opts[":getRowId"]}"')


        # Allow user-supplied extra grid options.
        opts.update(self._grid_config.extra_grid_options)

        
        return opts

    # ------------------------------------------------------------------
    # Internal: NiceGUI event handlers
    # ------------------------------------------------------------------

    def _on_cell_value_changed(self, e: events.GenericEventArguments) -> None:
        """Handle AG Grid ``cellValueChanged`` events from NiceGUI.

        This updates the internal row representation and calls any registered
        cell-edited handlers.

        Args:
            e: NiceGUI event arguments instance.
        """
        args: dict[str, Any] = e.args

        row_index = args.get("rowIndex")
        field = args.get("colId")
        old_value = args.get("oldValue")
        new_value = args.get("newValue")
        row_data = args.get("data") or {}

        if row_index is None or field is None:
            return

        i = int(row_index)
        if not (0 <= i < len(self._rows)):
            return

        self._rows[i][field] = new_value

        logger.debug(
            f"Cell edited: row={i}, field={field}, "
            f"old_value={old_value}, new_value={new_value}"
        )

        # Call registered handlers
        for handler in list(self._cell_edited_handlers):
            try:
                handler(i, field, old_value, new_value, row_data)
            except Exception:
                logger.exception("Error in cell_edited handler")

    def _on_row_selected(self, e: events.GenericEventArguments) -> None:
        args: dict[str, Any] = e.args

        logger.info('xxx')
        from pprint import pprint
        pprint(args)

        # 1) Prevent feedback loops from programmatic selection
        origin = getattr(self, "_selection_origin", "internal")
        if origin != "internal":
            # ignore events caused by our own set_selected_row_ids(...)
            return

        # 2) NiceGUI payload often looks like "rowClicked" and has no "selected"
        row_id = args.get("rowId")
        row_index = args.get("rowIndex")
        row_data = args.get("data") or {}

        if row_id is None and row_index is None:
            return

        # 3) De-dupe repeated events for the same row
        if row_id is not None:
            if getattr(self, "_last_row_id", None) == row_id:
                return
            self._last_row_id = row_id

        if row_index is None:
            # we still want to store the row data if available
            row_index = 0
        i = int(row_index)

        self._last_selected_rows = [row_data]

        for handler in list(self._row_selected_handlers):
            try:
                handler(i, row_data)
            except Exception:
                logger.exception("Error in row_selected handler")

    def _on_cell_key_down(self, e: events.GenericEventArguments) -> None:
        """Handle keyboard events inside grid cells (for row navigation).

        Currently logs event args to discover the payload structure.
        Will be refined to handle ArrowUp/ArrowDown for row selection once
        we see the actual event structure.
        """
        from pprint import pformat
        logger.info(f"AGGrid cellKeyDown event args:\n{pformat(e.args)}")
        # For now, do nothing else. We'll refine once we see what e.args looks like.

    def set_selected_row_ids(self, row_ids: list[str], *, origin: str = "external") -> None:
        """Programmatically select rows by row id.

        Args:
            row_ids: Row IDs to select. Requires GridConfig.row_id_field.
            origin: Tag used to prevent feedback loops (e.g. 'external').

        Raises:
            ValueError: if row_id_field is not configured.
        """
        if not self._grid_config.row_id_field:
            raise ValueError("GridConfig.row_id_field is required for programmatic selection")

        # We temporarily mark origin so rowSelected events caused by setSelected don't re-trigger handlers.
        self._selection_origin = origin

        # Clear selection first for single-select semantics
        if self._grid_config.selection_mode in {"single", "none"}:
            self._grid.run_grid_method("deselectAll")

        for i, rid in enumerate(row_ids):
            clear = True if (i == 0 and self._grid_config.selection_mode == "single") else False
            self._grid.run_row_method(rid, "setSelected", True, clear)

        self._selection_origin = "internal"
