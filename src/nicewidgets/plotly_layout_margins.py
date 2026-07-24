"""Optional fixed layout-margin profiles for aligned Plotly figure stacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PlotlyLayoutMarginsProfile:
    """Fixed Plotly ``layout.margin`` values for a two-state axis-label toggle.

    Args:
        with_axis_labels: Margins when axis decorations are visible.
        compact: Margins when axis decorations are hidden.
        stabilize_axis_automargin: When ``True``, force ``automargin=False`` on
            primary x/y axes so tick-label width does not shift the plot area.
    """

    with_axis_labels: dict[str, int]
    compact: dict[str, int]
    stabilize_axis_automargin: bool = False

    def resolve(self, *, show_axis_labels: bool) -> dict[str, int]:
        """Return a margin dictionary for the current axis-label state.

        Args:
            show_axis_labels: Whether axis decorations are visible.

        Returns:
            Plotly ``layout.margin`` dictionary with ``l``, ``r``, ``t``, and ``b``.
        """
        source = self.with_axis_labels if show_axis_labels else self.compact
        return dict(source)

    def apply_axis_stabilization(self, layout: dict[str, Any]) -> None:
        """Disable axis automargin on primary cartesian axes when configured.

        Args:
            layout: Plotly figure ``layout`` dictionary to update in place.

        Returns:
            None.
        """
        if not self.stabilize_axis_automargin:
            return
        for axis_name in ("xaxis", "yaxis"):
            axis = layout.setdefault(axis_name, {})
            if isinstance(axis, dict):
                axis["automargin"] = False
