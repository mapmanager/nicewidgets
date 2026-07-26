"""Shared AG Grid Enterprise module configuration.

NiceGUI loads AG Grid Enterprise as an optional browser-side ESM module.
Using this URL enables enterprise features but does not provide an AG Grid
license; production host applications remain responsible for licensing.
"""

DEFAULT_AG_GRID_ENTERPRISE_MODULE_URL = (
    'https://cdn.jsdelivr.net/npm/ag-grid-enterprise@34.2.0/+esm'
)
