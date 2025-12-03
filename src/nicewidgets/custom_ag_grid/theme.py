from __future__ import annotations

from nicegui import ui


# AG Grid theme CSS with optional behavior controlled via container classes:
# - .aggrid-zebra   → zebra rows
# - .aggrid-hover   → row hover highlight
# - .aggrid-tight   → tighter padding + smaller font
_THEME_CSS = """
<style>
/* Zebra rows (only when parent has .aggrid-zebra class) */
.aggrid-zebra .ag-row-even .ag-cell {
    background-color: #f7f7f7;
}
.aggrid-zebra .ag-row-odd .ag-cell {
    background-color: #ffffff;
}

/* Hover highlight (only when parent has .aggrid-hover class) */
.aggrid-hover .ag-row-hover .ag-cell {
    background-color: #e8f3ff;
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
