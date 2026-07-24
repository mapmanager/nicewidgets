"""Tests for Plotly trace overlays."""

from __future__ import annotations

from nicewidgets.raster_viewer.frontend.trace_overlay import (
    TRACE_OVERLAY_ID_META_KEY,
    TRACE_OVERLAY_META_KEY,
    TRACE_OVERLAY_META_TYPE,
    PlotlyTraceOverlay,
    PlotlyTraceOverlayLayer,
)


def test_trace_overlay_defaults_to_scattergl_markers() -> None:
    """Trace overlays should default to WebGL marker rendering."""
    layer = PlotlyTraceOverlayLayer()
    layer.add_overlay(PlotlyTraceOverlay(trace_id='left', x=[0.0, 1.0], y=[2.0, 3.0]))

    traces = layer.to_traces()

    assert len(traces) == 1
    assert traces[0]['type'] == 'scattergl'
    assert traces[0]['mode'] == 'markers'


def test_trace_overlay_preserves_metadata() -> None:
    """Trace overlays should include metadata identifying managed traces."""
    layer = PlotlyTraceOverlayLayer()
    layer.add_overlay(PlotlyTraceOverlay(trace_id='right', x=[0.0], y=[1.0]))

    trace = layer.to_traces()[0]

    assert trace['meta'] == {
        TRACE_OVERLAY_META_KEY: TRACE_OVERLAY_META_TYPE,
        TRACE_OVERLAY_ID_META_KEY: 'right',
    }
    assert PlotlyTraceOverlayLayer.is_trace_overlay(trace)
    assert PlotlyTraceOverlayLayer.trace_id_from_trace(trace) == 'right'


def test_trace_overlay_allows_explicit_scatter_type() -> None:
    """Trace overlays should allow callers to request SVG scatter traces."""
    layer = PlotlyTraceOverlayLayer()
    layer.add_overlay(
        PlotlyTraceOverlay(
            trace_id='small-overlay',
            x=[0.0],
            y=[1.0],
            plotly_type='scatter',
        )
    )

    trace = layer.to_traces()[0]

    assert trace['type'] == 'scatter'
    assert trace['mode'] == 'markers'


def test_empty_trace_overlay_layer_returns_no_traces() -> None:
    """Empty overlay layers should emit no Plotly traces."""
    layer = PlotlyTraceOverlayLayer()

    assert layer.to_traces() == []
