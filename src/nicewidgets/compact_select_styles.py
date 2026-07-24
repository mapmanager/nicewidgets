"""Compact Quasar QSelect styling for toolbar controls only.

Inject :func:`ensure_compact_select_styles` once per page build, then apply
:data:`COMPACT_SELECT_CLASS` and :data:`COMPACT_SELECT_PROPS` to individual
``ui.select`` instances. This is intentionally **not** wired into
:func:`nicewidgets.gui_defaults.setUpGuiDefaults` so schema cards, analysis
forms, and other selects keep standard QField height.
"""

from __future__ import annotations

from nicegui import ui

COMPACT_SELECT_CLASS = 'nw-select-compact'

# Quasar props for toolbar/contrast selects (see QSelect API: dense, hide-bottom-space, options-dense).
COMPACT_SELECT_PROPS = 'standout dense hide-bottom-space options-dense'

_COMPACT_SELECT_CSS = """
<style>
.nw-select-compact,
.nw-select-compact .q-field__control,
.nw-select-compact .q-field__append,
.nw-select-compact .q-field__control--addon {
    height: 30px !important;
    max-height: 30px !important;
    min-height: 30px !important;
    align-items: center;
}
.nw-select-compact .q-field__control-container {
    display: flex;
    align-items: center;
}
</style>
"""

_INJECTED = False


def ensure_compact_select_styles() -> None:
    """Inject compact-select CSS once per process (idempotent).

    Safe to call from every toolbar widget constructor; only the first call
    registers the shared stylesheet.
    """
    global _INJECTED
    if not _INJECTED:
        ui.add_head_html(_COMPACT_SELECT_CSS, shared=True)
        _INJECTED = True
