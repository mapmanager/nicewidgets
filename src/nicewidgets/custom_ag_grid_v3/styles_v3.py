# ../nicewidgets/src/nicewidgets/custom_ag_grid_v3/styles_v3.py
# v3-only CSS.
#
# Goal: hide AG Grid focused-cell visuals WITHOUT disabling focus (keyboard nav must keep working).
# Theme note: NiceGUI often applies `ag-theme-quartz` to an INNER div, not the outer ui.aggrid element.
# So use descendant selectors: `.kym-aggrid-v3 .ag-root ...`
#
# How to check theme (native-safe): print/log `grid.grid.theme` -> e.g. 'quartz'

from __future__ import annotations

from nicegui import ui


def install_styles_v3() -> None:
    ui.add_css(
        r"""
/* ---- Theme variables (safe, no behavior change) ---- */
.kym-aggrid-v3 {
  --ag-focus-shadow: none;
  --ag-range-selection-border-color: transparent;
  --ag-range-selection-border-style: none;
  --ag-range-selection-highlight-color: transparent;
}

/* ============================================================
   Focus box removal (DESCENDANT selectors; works with Quartz DOM)
   Based on the proven v2 pattern: target inside `.ag-root`.
   ============================================================ */

/* Hide focus outline on focused cell */
.kym-aggrid-v3 .ag-root .ag-cell:focus {
  outline: none !important;
}

/* Common focus ring classes across themes/versions */
.kym-aggrid-v3 .ag-root .ag-cell-focus,
.kym-aggrid-v3 .ag-root .ag-cell-focus:not(.ag-cell-range-selected),
.kym-aggrid-v3 .ag-root .ag-cell-focus:not(.ag-cell-range-selected):focus-within,
.kym-aggrid-v3 .ag-root .ag-cell-focus.ag-cell {
  outline: none !important;
  box-shadow: none !important;
  border-color: transparent !important;
  border: none !important;
}

/* Many themes draw focus via ::after */
.kym-aggrid-v3 .ag-root .ag-cell-focus::after,
.kym-aggrid-v3 .ag-root .ag-cell-focus:after,
.kym-aggrid-v3 .ag-root .ag-cell-focus:not(.ag-cell-range-selected)::after,
.kym-aggrid-v3 .ag-root .ag-cell-focus:not(.ag-cell-range-selected):after {
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}

/* Some builds use a dedicated "focus border" element/class */
.kym-aggrid-v3 .ag-root .ag-focus-managed:focus {
  outline: none !important;
  box-shadow: none !important;
}

/* ---- Range selection visuals (keep behavior, hide paint) ---- */
.kym-aggrid-v3 .ag-root .ag-cell-range-selected,
.kym-aggrid-v3 .ag-root .ag-cell-range-selected-1,
.kym-aggrid-v3 .ag-root .ag-cell-range-selected-2,
.kym-aggrid-v3 .ag-root .ag-cell-range-selected-3,
.kym-aggrid-v3 .ag-root .ag-cell-range-selected-4 {
  background: inherit !important;
}

/* ---- Header focus visuals ---- */
.kym-aggrid-v3 .ag-root .ag-header-cell:focus,
.kym-aggrid-v3 .ag-root .ag-header-cell-focus,
.kym-aggrid-v3 .ag-root .ag-header-cell-focus::after,
.kym-aggrid-v3 .ag-root .ag-header-cell-focus:after {
  outline: none !important;
  box-shadow: none !important;
  border: none !important;
}
        """
    )