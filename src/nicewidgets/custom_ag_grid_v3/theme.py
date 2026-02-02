from __future__ import annotations

from nicegui import ui


# AG Grid theme CSS with optional behavior controlled via container classes:
# - .aggrid-zebra   → zebra rows
# - .aggrid-hover   → row hover highlight
# - .aggrid-no-hover → remove row hover highlight (override theme default)
# - .aggrid-tight   → tighter padding + smaller font
# NOTE: zebra/hover rules avoid `.ag-row-selected` so the built-in blue selection
# styling from the AG Grid theme remains visible even when these helpers apply.
_THEME_CSS = """
<style>
/* Zebra rows (scoped to grid element) */
.aggrid-scope.aggrid-zebra .ag-row-even:not(.ag-row-selected) .ag-cell {
    background-color: #f7f7f7;
}
.aggrid-scope.aggrid-zebra .ag-row-odd:not(.ag-row-selected) .ag-cell {
    background-color: #ffffff;
}
/* Keep text readable when zebra/hover backgrounds apply */
.aggrid-zebra .ag-cell,
.aggrid-hover .ag-cell {
    color: var(--ag-foreground-color, inherit);
}

/* Hover highlight (only when parent has .aggrid-hover class) */
.aggrid-hover .ag-row-hover:not(.ag-row-selected) .ag-cell {
    background-color: #e8f3ff;
}

/* Disable hover highlight (override theme default hover styling) */
.aggrid-no-hover .ag-row-hover .ag-cell,
.aggrid-no-hover .ag-row-hover {
    background-color: inherit;
}

/* Tighter layout (only when parent has .aggrid-tight class) */
.aggrid-tight .ag-cell,
.aggrid-tight .ag-header-cell {
    padding: 2px 6px;
    font-size: 0.80rem;
    line-height: 1.2;
}
</style>
"""

_theme_injected: bool = False


def ensure_aggrid_theme() -> None:
    """Inject the AG Grid CSS theme once per application.

    This function is idempotent: calling it multiple times is safe and
    will only add the CSS block to the document head once.
    """
    global _theme_injected
    if not _theme_injected:
        ui.add_head_html(_THEME_CSS)
        _theme_injected = True
