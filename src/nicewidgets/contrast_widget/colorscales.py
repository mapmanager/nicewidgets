"""Utilities for managing Plotly colorscales.

Note: Single source of truth is kymflow.core.plotting.colorscales.
Sync or review when kymflow plotting changes.
"""

from __future__ import annotations

from typing import Dict, List

# Common Plotly colorscale options
COLORSCALE_OPTIONS: List[Dict[str, str]] = [
    {"label": "Grayscale", "value": "Gray"},
    {"label": "Grayscale (Inverted)", "value": "inverted_grays"},
    {"label": "Viridis", "value": "Viridis"},
    {"label": "Plasma", "value": "Plasma"},
    {"label": "Hot", "value": "Hot"},
    {"label": "Jet", "value": "Jet"},
    {"label": "Cool", "value": "Cool"},
    {"label": "Rainbow", "value": "Rainbow"},
]
