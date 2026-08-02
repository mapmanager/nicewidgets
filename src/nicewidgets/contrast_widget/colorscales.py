"""Color lookup table options for the contrast widget.

Maps human-friendly LUT names to Plotly colorscale strings or custom stop
lists. Channel-friendly named scales (``Red`` / ``Green`` / ``Blue`` /
``Yellow`` / ``Magenta``) are used both for single-channel heatmaps and for
multi-channel composite RGB sampling.
"""

from __future__ import annotations

COLORSCALE_OPTIONS: list[dict[str, str]] = [
    {'label': 'Gray', 'value': 'Gray'},
    {'label': 'Inverted Gray', 'value': 'inverted_grays'},
    {'label': 'Viridis', 'value': 'Viridis'},
    {'label': 'Plasma', 'value': 'Plasma'},
    {'label': 'Hot', 'value': 'Hot'},
    {'label': 'Jet', 'value': 'Jet'},
    {'label': 'Cool', 'value': 'Cool'},
    {'label': 'Rainbow', 'value': 'Rainbow'},
    {'label': 'Red', 'value': 'Red'},
    {'label': 'Green', 'value': 'Green'},
    {'label': 'Blue', 'value': 'Blue'},
    {'label': 'Yellow', 'value': 'Yellow'},
    {'label': 'Magenta', 'value': 'Magenta'},
]


def colorscale_option_values() -> list[str]:
    """Return the ordered list of LUT identifier strings used by the widget.

    Returns:
        Ordered list of ``COLORSCALE_OPTIONS`` ``value`` strings.
    """
    return [opt['value'] for opt in COLORSCALE_OPTIONS]


def colorscale_option_value_to_label() -> dict[str, str]:
    """Return ``{value: label}`` mapping suitable for NiceGUI ``ui.select``.

    The wire value (what handlers receive and what callers store) remains the
    internal identifier such as ``'inverted_grays'``; only the displayed label
    differs (``'Inverted Gray'``).

    Returns:
        Ordered mapping from LUT value to user-facing label.
    """
    return {opt['value']: opt['label'] for opt in COLORSCALE_OPTIONS}


def get_colorscale(name: str) -> str | list[list[float | str]]:
    """Map a widget LUT identifier to a Plotly-compatible colorscale value.

    Args:
        name: LUT identifier from :data:`COLORSCALE_OPTIONS` values.

    Returns:
        Plotly colorscale string for built-in scales, or a stop list for custom
        LUTs. Unknown names are returned unchanged so downstream callers may
        pass through other Plotly colorscale names.
    """
    if name == 'Gray':
        return 'Greys'
    if name == 'inverted_grays':
        return [[0, 'rgb(255,255,255)'], [1, 'rgb(0,0,0)']]
    if name == 'Red':
        return 'Reds'
    if name == 'Green':
        return 'Greens'
    if name == 'Blue':
        return 'Blues'
    # Multi-stop channel LUTs aligned with acqstore demo composite coloring.
    if name == 'Yellow':
        return [
            [0.0, 'rgb(0,0,0)'],
            [0.35, 'rgb(105,80,0)'],
            [0.75, 'rgb(255,210,0)'],
            [1.0, 'rgb(255,255,220)'],
        ]
    if name == 'Magenta':
        return [
            [0.0, 'rgb(0,0,0)'],
            [0.35, 'rgb(95,0,105)'],
            [0.75, 'rgb(255,0,220)'],
            [1.0, 'rgb(255,230,255)'],
        ]
    return name
