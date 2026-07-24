"""Plotly x/y trace overlay helpers for raster viewer widgets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

TRACE_OVERLAY_META_TYPE = 'trace'
TRACE_OVERLAY_META_KEY = 'nicewidgets_overlay_type'
TRACE_OVERLAY_ID_META_KEY = 'trace_id'


@dataclass(frozen=True, slots=True)
class PlotlyTraceOverlay:
    """Marker trace overlay in Plotly physical coordinates.

    Args:
        trace_id: Stable overlay identifier.
        x: Plotly x coordinates in physical units.
        y: Plotly y coordinates in physical units.
        color: Optional marker and line color when the trace mode includes
            markers and/or lines.
        line_width: Optional line width in pixels when the trace mode includes
            lines.
        name: Optional Plotly trace display name.
        visible: Whether the trace is visible.
        plotly_type: Plotly trace type used for rendering. Defaults to ``scattergl``
            for large browser-side point overlays.
        mode: Plotly trace mode. Defaults to ``markers``.
    """

    trace_id: str
    x: Sequence[float]
    y: Sequence[float]
    color: str | None = None
    line_width: float | None = None
    name: str | None = None
    visible: bool = True
    plotly_type: Literal['scatter', 'scattergl'] = 'scattergl'
    mode: Literal['markers', 'lines', 'lines+markers'] = 'markers'


class PlotlyTraceOverlayLayer:
    """Manage marker trace overlays in a Plotly ``data`` list.

    This class owns trace overlay state and converts that state to Plotly trace
    dictionaries. It does not call NiceGUI or Plotly directly; callers mutate
    their figure dict and push overlay traces with browser-side Plotly APIs.
    """

    def __init__(self) -> None:
        self._overlays: dict[str, PlotlyTraceOverlay] = {}

    @property
    def overlays(self) -> tuple[PlotlyTraceOverlay, ...]:
        """Return managed overlays in insertion order."""
        return tuple(self._overlays.values())

    @staticmethod
    def _normalized_trace_id(trace_id: str) -> str:
        normalized = str(trace_id)
        if not normalized:
            raise ValueError('Trace overlay trace_id must be non-empty.')
        return normalized

    @classmethod
    def _validate_overlay(cls, overlay: PlotlyTraceOverlay) -> PlotlyTraceOverlay:
        cls._normalized_trace_id(overlay.trace_id)
        if len(overlay.x) != len(overlay.y):
            raise ValueError('Trace overlay x and y must have the same length.')
        return overlay

    @staticmethod
    def trace_id_from_trace(trace: dict[str, object]) -> str | None:
        """Return trace id encoded in a Plotly trace dictionary."""
        meta = trace.get('meta')
        if not isinstance(meta, dict):
            return None
        if meta.get(TRACE_OVERLAY_META_KEY) != TRACE_OVERLAY_META_TYPE:
            return None
        trace_id = meta.get(TRACE_OVERLAY_ID_META_KEY)
        return trace_id if isinstance(trace_id, str) and trace_id else None

    @classmethod
    def is_trace_overlay(cls, trace: object) -> bool:
        """Return whether an object is a managed trace overlay."""
        return isinstance(trace, dict) and cls.trace_id_from_trace(trace) is not None

    def set_overlays(self, overlays: Iterable[PlotlyTraceOverlay]) -> None:
        """Replace all managed trace overlays."""
        next_overlays: dict[str, PlotlyTraceOverlay] = {}
        for overlay in overlays:
            self._validate_overlay(overlay)
            next_overlays[self._normalized_trace_id(overlay.trace_id)] = overlay
        self._overlays = next_overlays

    def add_overlay(self, overlay: PlotlyTraceOverlay) -> None:
        """Add or replace one trace overlay."""
        self._validate_overlay(overlay)
        self._overlays[self._normalized_trace_id(overlay.trace_id)] = overlay

    def delete_overlay(self, trace_id: str) -> None:
        """Delete one trace overlay."""
        self._overlays.pop(self._normalized_trace_id(trace_id), None)

    def clear_overlays(self) -> None:
        """Delete all trace overlays."""
        self._overlays.clear()

    def merge_traces(self, existing_traces: Iterable[object]) -> list[dict[str, object]]:
        """Return non-overlay traces plus current overlay traces."""
        merged: list[dict[str, object]] = [
            dict(trace)
            for trace in existing_traces
            if isinstance(trace, dict) and not self.is_trace_overlay(trace)
        ]
        merged.extend(self.to_traces())
        return merged

    def to_traces(self) -> list[dict[str, object]]:
        """Return current overlays as Plotly scatter trace dictionaries."""
        return [self._overlay_to_trace(overlay) for overlay in self._overlays.values()]

    def _overlay_to_trace(self, overlay: PlotlyTraceOverlay) -> dict[str, object]:
        """Convert one overlay to a Plotly marker trace dictionary."""
        trace: dict[str, object] = {
            'type': overlay.plotly_type,
            'mode': overlay.mode,
            'x': [float(value) for value in overlay.x],
            'y': [float(value) for value in overlay.y],
            'name': overlay.name or overlay.trace_id,
            'visible': bool(overlay.visible),
            'meta': {
                TRACE_OVERLAY_META_KEY: TRACE_OVERLAY_META_TYPE,
                TRACE_OVERLAY_ID_META_KEY: overlay.trace_id,
            },
        }
        if overlay.color is not None:
            trace['marker'] = {'color': overlay.color}
        if 'line' in overlay.mode:
            line_style: dict[str, object] = {}
            if overlay.color is not None:
                line_style['color'] = overlay.color
            if overlay.line_width is not None:
                line_style['width'] = overlay.line_width
            if line_style:
                trace['line'] = line_style
        return trace
