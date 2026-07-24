"""User-intent payload emitted by ``ContrastWidget`` (frozen dataclass, no host-application imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContrastChangedIntent:
    """User changed any contrast control (LUT, range, or Auto).

    The widget always emits the full current state, regardless of which control
    the user touched. ``Auto`` button clicks are computed locally by the widget
    and emitted as a normal change.

    Args:
        color_lut: Selected LUT identifier from
            :data:`nicewidgets.contrast_widget.colorscales.COLORSCALE_OPTIONS`.
        value_min: Current minimum intensity in the displayed range.
        value_max: Current maximum intensity in the displayed range.
        from_auto: ``True`` when emitted by the Auto button; ``False`` for
            manual LUT or range slider edits.
    """

    color_lut: str
    value_min: int
    value_max: int
    from_auto: bool = False
