"""Tests for Plotly x/y trace overlay helpers."""

from __future__ import annotations

import pytest

from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer
from nicewidgets.raster_viewer.frontend.trace_overlay import (
    PlotlyTraceOverlay,
    PlotlyTraceOverlayLayer,
)


def test_trace_overlay_accepts_empty_xy() -> None:
    """Empty overlays are valid when x and y are both empty."""
    layer = PlotlyTraceOverlayLayer()
    layer.add_overlay(PlotlyTraceOverlay(trace_id='diameter', x=[], y=[]))

    traces = layer.to_traces()

    assert traces[0]['x'] == []
    assert traces[0]['y'] == []
    assert traces[0]['mode'] == 'markers'


def test_trace_overlay_rejects_empty_trace_id() -> None:
    """Trace overlay ids must be stable non-empty strings."""
    layer = PlotlyTraceOverlayLayer()

    with pytest.raises(ValueError, match='trace_id'):
        layer.add_overlay(PlotlyTraceOverlay(trace_id='', x=[], y=[]))


def test_trace_overlay_rejects_mismatched_xy_lengths() -> None:
    """Trace overlay x and y values must have matching lengths."""
    layer = PlotlyTraceOverlayLayer()

    with pytest.raises(ValueError, match='same length'):
        layer.add_overlay(PlotlyTraceOverlay(trace_id='diameter', x=[1.0], y=[]))


def test_trace_overlay_plotly_trace_has_metadata_and_optional_style() -> None:
    """Generated Plotly traces should be marker traces owned by metadata."""
    layer = PlotlyTraceOverlayLayer()
    layer.add_overlay(
        PlotlyTraceOverlay(
            trace_id='diameter',
            x=[1, 2],
            y=[3, 4],
            color='red',
            line_width=5.0,
            name='Diameter',
            visible=False,
        )
    )

    trace = layer.to_traces()[0]

    assert trace['type'] == 'scattergl'
    assert trace['mode'] == 'markers'
    assert trace['x'] == [1.0, 2.0]
    assert trace['y'] == [3.0, 4.0]
    assert trace['name'] == 'Diameter'
    assert trace['visible'] is False
    assert trace['marker'] == {'color': 'red'}
    assert 'line' not in trace
    assert trace['meta'] == {
        'nicewidgets_overlay_type': 'trace',
        'trace_id': 'diameter',
    }


def test_trace_overlay_lines_mode_emits_line_style() -> None:
    """Line modes should emit Plotly line styling alongside marker color."""
    layer = PlotlyTraceOverlayLayer()
    layer.add_overlay(
        PlotlyTraceOverlay(
            trace_id='scan_path',
            x=[0.0, 1.0],
            y=[2.0, 3.0],
            color='cyan',
            line_width=2.0,
            mode='lines+markers',
            plotly_type='scattergl',
        )
    )

    trace = layer.to_traces()[0]

    assert trace['mode'] == 'lines+markers'
    assert trace['marker'] == {'color': 'cyan'}
    assert trace['line'] == {'color': 'cyan', 'width': 2.0}


def test_trace_overlay_add_replaces_duplicate_and_delete_removes() -> None:
    """Adding duplicate trace ids should replace and delete should remove."""
    layer = PlotlyTraceOverlayLayer()
    layer.add_overlay(PlotlyTraceOverlay(trace_id='diameter', x=[1], y=[2]))
    layer.add_overlay(PlotlyTraceOverlay(trace_id='diameter', x=[3], y=[4]))

    assert len(layer.to_traces()) == 1
    assert layer.to_traces()[0]['x'] == [3.0]

    layer.delete_overlay('diameter')
    assert layer.to_traces() == []


def test_trace_overlay_merge_preserves_non_overlay_traces() -> None:
    """Trace replacement should keep raster and unrelated Plotly traces."""
    layer = PlotlyTraceOverlayLayer()
    layer.set_overlays([PlotlyTraceOverlay(trace_id='diameter', x=[1], y=[2])])

    merged = layer.merge_traces([
        {'type': 'image', 'name': 'raster'},
        {'type': 'scatter', 'name': 'crosshair'},
        {
            'type': 'scatter',
            'name': 'old overlay',
            'meta': {'nicewidgets_overlay_type': 'trace', 'trace_id': 'old'},
        },
    ])

    assert [trace['name'] for trace in merged] == ['raster', 'crosshair', 'diameter']


def test_plotly_raster_viewer_trace_overlay_api_mutates_data_without_plot_update() -> None:
    """Viewer trace APIs should mutate data without requiring a Plotly element."""
    viewer = PlotlyRasterViewer()
    viewer.figure['data'] = [
        {'type': 'image', 'name': 'raster'},
        {'type': 'scatter', 'name': 'crosshair'},
    ]

    viewer.set_trace_overlays([PlotlyTraceOverlay(trace_id='diameter', x=[1], y=[2])])
    assert [trace['name'] for trace in viewer.figure['data']] == ['raster', 'crosshair', 'diameter']

    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='diameter', x=[3], y=[4], name='new diameter'))
    assert [trace['name'] for trace in viewer.figure['data']] == ['raster', 'crosshair', 'new diameter']

    viewer.delete_trace_overlay('diameter')
    assert [trace['name'] for trace in viewer.figure['data']] == ['raster', 'crosshair']

    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='diameter', x=[], y=[]))
    viewer.clear_trace_overlays()
    assert [trace['name'] for trace in viewer.figure['data']] == ['raster', 'crosshair']
