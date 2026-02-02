# src/nicewidgets/custom_ag_grid/styles.py
from __future__ import annotations


AGGRID_HIDE_CELL_FOCUS_CSS = """
/* Hide AG Grid focused-cell visuals without disabling focus (keeps keyboard navigation working) */
.ag-root .ag-cell:focus {
  outline: none !important;
}

/* Different themes/versions use these classes for the focus ring */
.ag-root .ag-cell-focus,
.ag-root .ag-cell-focus:not(.ag-cell-range-selected),
.ag-root .ag-cell-focus:not(.ag-cell-range-selected):focus-within {
  outline: none !important;
  border: none !important;
  box-shadow: none !important;
}

.ag-root .ag-cell-focus.ag-cell {
  outline: none !important;
  box-shadow: none !important;
}
""".strip()