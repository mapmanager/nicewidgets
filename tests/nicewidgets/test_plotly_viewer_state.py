"""Headless state/math tests for PlotlyRasterViewer.

These tests avoid building the NiceGUI element (``ui.plotly``) and focus on
state transitions, helpers, and Plotly-dict mutations that can be exercised
without a live browser session.
"""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np
import pytest


if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.ui = types.SimpleNamespace()
    fake_nicegui.app = types.SimpleNamespace(native=types.SimpleNamespace(main_window=None))
    sys.modules['nicegui'] = fake_nicegui

from nicewidgets.raster_viewer.backend.image_model import (
    RasterGridSpec,
    RowColBounds,
)
from nicewidgets.raster_viewer.frontend.plotly_coord_transform import (
    PlotlyCoordTransform,
    merge_partial_relayout,
)
from nicewidgets.raster_viewer.frontend.plotly_protocol import (
    DEFAULT_HEATMAP_COLORSCALE,
    RASTER_VIEWER_PLOTLY_CONFIG,
    PlotlyViewportPayload,
    build_plotly_figure,
    parse_relayout_payload,
)
from nicewidgets.raster_viewer.frontend import plotly_viewer as plotly_viewer_module
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer
from nicewidgets.raster_viewer.frontend.roi_overlay import RectRoiOverlay
from nicewidgets.raster_viewer.frontend.trace_overlay import PlotlyTraceOverlay


def _grid() -> RasterGridSpec:
    return RasterGridSpec(dx=2.0, dy=4.0, x_unit='s', y_unit='um')


def _viewer_with_data(shape: tuple[int, int] = (10, 8)) -> PlotlyRasterViewer:
    """Build a viewer with data set (no NiceGUI element)."""
    viewer = PlotlyRasterViewer()
    data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    asyncio.run(viewer.set_data(data, grid=_grid()))
    return viewer


# ---------- PlotlyCoordTransform pure math ----------


def test_plot_xy_ranges_to_row_col_round_trip() -> None:
    """Plot coords should round-trip through ``plot_xy_ranges_to_row_col``."""
    grid = _grid()
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=grid)
    bounds = RowColBounds(row_min=2.0, row_max=8.0, col_min=1.0, col_max=6.0)
    x_lo, x_hi = t.row_col_to_plot_x_range(bounds)
    y_lo, y_hi = t.row_col_to_plot_y_range(bounds)

    back = t.plot_xy_ranges_to_row_col(x_lo, x_hi, y_lo, y_hi)

    assert back.row_min == pytest.approx(2.0)
    assert back.row_max == pytest.approx(8.0)
    assert back.col_min == pytest.approx(1.0)
    assert back.col_max == pytest.approx(6.0)


def test_row_col_to_plot_y_range_is_bottom_up() -> None:
    """Y-axis layout ranges should be ordered low to high for bottom-left origin."""
    grid = _grid()
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=grid)
    bounds = RowColBounds(row_min=0.0, row_max=10.0, col_min=1.0, col_max=6.0)

    assert t.row_col_to_plot_y_range(bounds) == (4.0, 24.0)


def test_plot_xy_ranges_to_row_col_clips_to_shape() -> None:
    """Out-of-range plot coords should be clipped to the source shape."""
    t = PlotlyCoordTransform(nrows=4, ncols=4, grid=_grid())
    back = t.plot_xy_ranges_to_row_col(-100.0, 100.0, -100.0, 100.0)

    assert back.row_min == 0.0
    assert back.row_max == 4.0
    assert back.col_min == 0.0
    assert back.col_max == 4.0


def test_full_row_col_bounds_returns_full_array() -> None:
    """``full_row_col_bounds`` should span the entire source array."""
    t = PlotlyCoordTransform(nrows=12, ncols=5, grid=_grid())
    b = t.full_row_col_bounds()

    assert b.row_min == 0.0
    assert b.row_max == 12.0
    assert b.col_min == 0.0
    assert b.col_max == 5.0


# ---------- merge_partial_relayout ----------


def test_merge_partial_relayout_fills_missing_y() -> None:
    """Missing y range should be filled from fallback bounds."""
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=_grid())
    fallback = RowColBounds(row_min=0, row_max=10, col_min=0, col_max=8)
    relayout = {'xaxis.range': [0.0, 5.0]}

    merged = merge_partial_relayout(relayout, t, fallback)

    assert 'xaxis.range' in merged
    assert 'yaxis.range' in merged


def test_merge_partial_relayout_fills_missing_x() -> None:
    """Missing x range should be filled from fallback bounds."""
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=_grid())
    fallback = RowColBounds(row_min=0, row_max=10, col_min=0, col_max=8)
    relayout = {'yaxis.range': [0.0, 10.0]}

    merged = merge_partial_relayout(relayout, t, fallback)

    assert 'xaxis.range' in merged
    assert 'yaxis.range' in merged


def test_merge_partial_relayout_keeps_existing_ranges() -> None:
    """Existing range keys should be preserved verbatim."""
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=_grid())
    fallback = RowColBounds(row_min=0, row_max=10, col_min=0, col_max=8)
    relayout = {
        'xaxis.range': [1.0, 2.0],
        'yaxis.range': [3.0, 4.0],
    }

    merged = merge_partial_relayout(relayout, t, fallback)

    assert merged['xaxis.range'] == [1.0, 2.0]
    assert merged['yaxis.range'] == [3.0, 4.0]


def test_merge_partial_relayout_accepts_bracket_keys() -> None:
    """Bracketed range keys (e.g. ``xaxis.range[0]``) should satisfy ``has_x``."""
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=_grid())
    fallback = RowColBounds(row_min=0, row_max=10, col_min=0, col_max=8)
    relayout = {'xaxis.range[0]': 0.5, 'xaxis.range[1]': 1.5}

    merged = merge_partial_relayout(relayout, t, fallback)

    assert 'xaxis.range[0]' in merged
    assert 'xaxis.range[1]' in merged
    assert 'yaxis.range' in merged
    assert 'xaxis.range' not in merged


# ---------- plotly_protocol helpers ----------


def test_raster_viewer_plotly_config_enables_doubleclick_reset() -> None:
    """Double-click must reach the custom overview-PNG reset handler."""
    assert RASTER_VIEWER_PLOTLY_CONFIG['doubleClick'] == 'reset'


def test_parse_relayout_payload_returns_view_request() -> None:
    """Parsing a relayout payload should return a ViewRequest with bounds and viewport."""
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=_grid())
    fallback = RowColBounds(row_min=0, row_max=10, col_min=0, col_max=8)
    payload = PlotlyViewportPayload(
        relayout={'xaxis.range': [0.0, 20.0], 'yaxis.range': [0.0, 32.0]},
        width_px=300,
        height_px=200,
    )

    req = parse_relayout_payload(payload, t, fallback)

    assert req.viewport.width_px == 300
    assert req.viewport.height_px == 200
    assert req.bounds.row_min == pytest.approx(0.0)
    assert req.bounds.row_max == pytest.approx(10.0)


def test_parse_relayout_payload_uses_fallback_for_missing_axis() -> None:
    """Missing axis keys should fall back to ``fallback_bounds``."""
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=_grid())
    fallback = RowColBounds(row_min=0, row_max=10, col_min=0, col_max=8)
    payload = PlotlyViewportPayload(
        relayout={},
        width_px=200,
        height_px=100,
    )

    req = parse_relayout_payload(payload, t, fallback)

    assert req.bounds.row_min == pytest.approx(0.0)
    assert req.bounds.row_max == pytest.approx(10.0)
    assert req.bounds.col_min == pytest.approx(0.0)
    assert req.bounds.col_max == pytest.approx(8.0)


def test_parse_relayout_payload_reads_bracket_keys() -> None:
    """``xaxis.range[0]`` / ``[1]`` keys should be honored when list key absent."""
    t = PlotlyCoordTransform(nrows=10, ncols=8, grid=_grid())
    fallback = RowColBounds(row_min=0, row_max=10, col_min=0, col_max=8)
    payload = PlotlyViewportPayload(
        relayout={
            'xaxis.range[0]': 0.0,
            'xaxis.range[1]': 4.0,
            'yaxis.range[0]': 0.0,
            'yaxis.range[1]': 16.0,
        },
        width_px=100,
        height_px=100,
    )

    req = parse_relayout_payload(payload, t, fallback)

    assert req.bounds.row_min == pytest.approx(0.0)
    assert req.bounds.row_max == pytest.approx(2.0)


def test_build_plotly_figure_image_mode() -> None:
    """``image_png`` mode should produce a Plotly image trace."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    response = RenderResponse(
        mode='image_png',
        level=0,
        bounds=RowColBounds(row_min=0, row_max=4, col_min=0, col_max=4),
        shape=(4, 4),
        grid=_grid(),
        x0=0.0,
        y0=0.0,
        dx=2.0,
        dy=4.0,
        png_data_uri='data:image/png;base64,AAA',
    )

    fig = build_plotly_figure(response, uirevision='ui-1')

    trace = fig['data'][0]
    assert trace['type'] == 'image'
    assert trace['source'] == 'data:image/png;base64,AAA'
    assert trace['x0'] == pytest.approx(1.0)
    assert trace['y0'] == pytest.approx(2.0)
    assert trace['dx'] == pytest.approx(2.0)
    assert trace['dy'] == pytest.approx(4.0)
    assert fig['layout']['xaxis']['range'] == [0.0, 8.0]
    assert fig['layout']['yaxis']['range'] == [0.0, 16.0]
    assert fig['layout']['uirevision'] == 'ui-1'


def test_build_plotly_figure_heatmap_mode() -> None:
    """``heatmap_z`` mode should produce a Plotly heatmap trace with z data."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    z = np.array([[1.0, 2.0], [3.0, 4.0]])
    response = RenderResponse(
        mode='heatmap_z',
        level=0,
        bounds=RowColBounds(row_min=0, row_max=2, col_min=0, col_max=2),
        shape=(2, 2),
        grid=_grid(),
        x0=0.0,
        y0=0.0,
        dx=2.0,
        dy=4.0,
        z=z,
        zmin=1.0,
        zmax=4.0,
    )

    fig = build_plotly_figure(response, heatmap_colorscale='Viridis')

    trace = fig['data'][0]
    assert trace['type'] == 'heatmap'
    assert trace['colorscale'] == 'Viridis'
    assert trace['z'] == [[1.0, 2.0], [3.0, 4.0]]
    assert trace['x0'] == pytest.approx(1.0)
    assert trace['y0'] == pytest.approx(2.0)
    assert trace['dx'] == pytest.approx(2.0)
    assert trace['dy'] == pytest.approx(4.0)
    assert fig['layout']['xaxis']['range'] == [0.0, 4.0]
    assert fig['layout']['yaxis']['range'] == [0.0, 8.0]


def test_build_plotly_figure_raster_origin_handles_non_unit_scale_and_offset() -> None:
    """Raster traces should use pixel centers while axes remain edge-based."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    grid = RasterGridSpec(dx=0.01, dy=0.25, x_unit='s', y_unit='um')
    bounds = RowColBounds(row_min=10, row_max=14, col_min=3, col_max=7)
    base_kwargs = dict(
        level=0,
        bounds=bounds,
        shape=(20, 10),
        grid=grid,
        x0=0.10,
        y0=0.75,
        dx=0.01,
        dy=0.25,
    )

    image_response = RenderResponse(
        mode='image_png',
        png_data_uri='data:image/png;base64,AAA',
        **base_kwargs,
    )
    heatmap_response = RenderResponse(
        mode='heatmap_z',
        z=np.ones((4, 4), dtype=np.float32),
        **base_kwargs,
    )

    image_trace = build_plotly_figure(image_response)['data'][0]
    heatmap_trace = build_plotly_figure(heatmap_response)['data'][0]

    for trace in (image_trace, heatmap_trace):
        assert trace['x0'] == pytest.approx(0.105)
        assert trace['y0'] == pytest.approx(0.875)
        assert trace['dx'] == pytest.approx(0.01)
        assert trace['dy'] == pytest.approx(0.25)

    fig = build_plotly_figure(image_response)
    assert fig['layout']['xaxis']['range'] == pytest.approx([0.10, 0.14])
    assert fig['layout']['yaxis']['range'] == pytest.approx([0.75, 1.75])


def test_build_plotly_figure_heatmap_requires_z() -> None:
    """Heatmap mode without z data should raise ``ValueError``."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    response = RenderResponse(
        mode='heatmap_z',
        level=0,
        bounds=RowColBounds(row_min=0, row_max=2, col_min=0, col_max=2),
        shape=(2, 2),
        grid=_grid(),
        x0=0.0,
        y0=0.0,
        dx=2.0,
        dy=4.0,
    )

    with pytest.raises(ValueError, match='numeric z data'):
        build_plotly_figure(response)


# ---------- PlotlyRasterViewer initial state ----------


def test_viewer_initial_state() -> None:
    """A freshly-constructed viewer should have no data, plot, or transform."""
    viewer = PlotlyRasterViewer()

    assert viewer.has_data is False
    assert viewer.plot is None
    assert isinstance(viewer.figure, dict)
    assert isinstance(viewer.current_bounds, RowColBounds)


def test_new_uirevision_returns_unique_strings() -> None:
    """``_new_uirevision`` should produce unique non-empty strings."""
    a = PlotlyRasterViewer._new_uirevision()
    b = PlotlyRasterViewer._new_uirevision()

    assert isinstance(a, str) and a
    assert isinstance(b, str) and b
    assert a != b


def test_build_initial_figure_no_data_returns_scaffold() -> None:
    """Without data, ``_build_initial_figure`` should return an empty scaffold."""
    viewer = PlotlyRasterViewer()
    fig = viewer._build_initial_figure()

    assert fig['data'] == []
    assert 'layout' in fig
    assert fig['layout']['autosize'] is True
    assert 'config' in fig


def test_display_style_uses_defaults() -> None:
    """``_display_style`` should reflect default colorscale and unset contrast."""
    viewer = PlotlyRasterViewer()
    style = viewer._display_style()

    assert style.colorscale == DEFAULT_HEATMAP_COLORSCALE
    assert style.zmin is None
    assert style.zmax is None


def test_js_plotly_graph_div_returns_early_without_plot() -> None:
    """Without a built plot element, the JS snippet should short-circuit."""
    viewer = PlotlyRasterViewer()

    assert viewer._js_plotly_graph_div() == 'return;'


def test_layout_pin_xy_ranges_writes_axis_ranges() -> None:
    """``_layout_pin_xy_ranges`` should pin axes in the plotly dict."""
    viewer = PlotlyRasterViewer()
    viewer._layout_pin_xy_ranges(x_lo=1.0, x_hi=5.0, y_lo=2.0, y_hi=10.0)

    layout = viewer._plotly_dict['layout']
    assert layout['xaxis']['range'] == [1.0, 5.0]
    assert layout['xaxis']['autorange'] is False
    assert layout['yaxis']['range'] == [2.0, 10.0]
    assert layout['yaxis']['autorange'] is False


def test_heatmap_and_image_trace_active_return_false_when_no_plot() -> None:
    """Both trace detection helpers should return False without a built plot."""
    viewer = PlotlyRasterViewer()

    assert viewer._heatmap_trace_active() is False
    assert viewer._image_trace_active() is False


# ---------- set_data side effects ----------


def test_set_data_populates_service_transform_and_bounds() -> None:
    """``set_data`` should configure service, transform, bounds, and revision."""
    viewer = _viewer_with_data(shape=(10, 8))

    assert viewer.has_data is True
    assert viewer._service is not None
    assert viewer._transform is not None
    assert viewer.current_bounds.row_min == 0.0
    assert viewer.current_bounds.row_max == 10.0
    assert viewer.current_bounds.col_min == 0.0
    assert viewer.current_bounds.col_max == 8.0


def test_set_data_resets_contrast_and_colorscale() -> None:
    """``set_data`` should reset contrast window and colorscale to defaults."""
    viewer = PlotlyRasterViewer()
    viewer._contrast_zmin = 0.1
    viewer._contrast_zmax = 0.9
    viewer._heatmap_colorscale = 'Viridis'

    asyncio.run(viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid()))

    assert viewer._contrast_zmin is None
    assert viewer._contrast_zmax is None
    assert viewer._heatmap_colorscale == DEFAULT_HEATMAP_COLORSCALE


def test_set_data_clears_existing_trace_overlays() -> None:
    """``set_data`` should clear previously-added trace overlays."""
    viewer = PlotlyRasterViewer()
    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='t', x=[0.0], y=[0.0]))

    asyncio.run(viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid()))

    figure_data = viewer.figure.get('data', [])
    assert isinstance(figure_data, list)
    assert len(figure_data) == 1  # only the raster trace remains


def test_set_data_then_build_initial_figure_returns_full_figure() -> None:
    """After ``set_data``, ``_build_initial_figure`` should include a raster trace."""
    viewer = _viewer_with_data(shape=(4, 4))

    fig = viewer._build_initial_figure()

    assert isinstance(fig['data'], list)
    assert len(fig['data']) == 1
    assert fig['data'][0]['type'] in {'image', 'heatmap'}


# ---------- ROI overlay state (no plot element) ----------


def test_set_rois_syncs_layout_shapes() -> None:
    """``set_rois`` should sync overlay rectangles into ``layout.shapes``."""
    viewer = PlotlyRasterViewer()
    roi = RectRoiOverlay(roi_id=1, x0=0.0, x1=2.0, y0=0.0, y1=4.0, label='r1')
    viewer.set_rois([roi])

    shapes = viewer._plotly_dict['layout']['shapes']
    assert len(shapes) == 1
    assert shapes[0]['name'] == 'roi:1'


def test_add_roi_appends_shape() -> None:
    """``add_roi`` should append a new shape into the plotly dict."""
    viewer = PlotlyRasterViewer()
    viewer.add_roi(RectRoiOverlay(roi_id=2, x0=0.0, x1=1.0, y0=0.0, y1=1.0))

    shapes = viewer._plotly_dict['layout']['shapes']
    assert any(s.get('name') == 'roi:2' for s in shapes)


def test_delete_roi_removes_shape() -> None:
    """``delete_roi`` should remove the matching shape from the plotly dict."""
    viewer = PlotlyRasterViewer()
    viewer.add_roi(RectRoiOverlay(roi_id=2, x0=0.0, x1=1.0, y0=0.0, y1=1.0))
    viewer.add_roi(RectRoiOverlay(roi_id=3, x0=0.0, x1=1.0, y0=0.0, y1=1.0))

    viewer.delete_roi(2)

    shapes = viewer._plotly_dict['layout']['shapes']
    names = {s.get('name') for s in shapes}
    assert 'roi:2' not in names
    assert 'roi:3' in names


def test_select_roi_marks_selection() -> None:
    """``select_roi`` should update the ROI layer's selection state."""
    viewer = PlotlyRasterViewer()
    viewer.add_roi(RectRoiOverlay(roi_id=7, x0=0.0, x1=1.0, y0=0.0, y1=1.0))

    viewer.select_roi(7)

    assert viewer._plotly_rois.selected_roi_id == 7


def test_sync_roi_shapes_handles_non_list_existing_shapes() -> None:
    """``_sync_roi_shapes_to_plotly_dict`` should tolerate a non-list ``shapes``."""
    viewer = PlotlyRasterViewer()
    viewer._plotly_dict['layout'] = {'shapes': 'not a list'}
    viewer.add_roi(RectRoiOverlay(roi_id=1, x0=0.0, x1=1.0, y0=0.0, y1=1.0))

    shapes = viewer._plotly_dict['layout']['shapes']
    assert isinstance(shapes, list)
    assert any(s.get('name') == 'roi:1' for s in shapes)


# ---------- Trace overlay state (no plot element) ----------


def test_set_trace_overlays_syncs_data_traces() -> None:
    """``set_trace_overlays`` should merge overlay traces into ``data``."""
    viewer = PlotlyRasterViewer()
    overlay = PlotlyTraceOverlay(trace_id='left', x=[0.0, 1.0], y=[2.0, 3.0])

    viewer.set_trace_overlays([overlay])

    data = viewer._plotly_dict.get('data', [])
    assert isinstance(data, list)
    assert len(data) >= 1


def test_delete_trace_overlay_removes_trace() -> None:
    """``delete_trace_overlay`` should remove the matching overlay."""
    viewer = PlotlyRasterViewer()
    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='a', x=[0.0], y=[0.0]))
    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='b', x=[0.0], y=[0.0]))

    viewer.delete_trace_overlay('a')

    overlays = viewer._plotly_trace_overlays
    ids = {o.trace_id for o in overlays.overlays}
    assert 'a' not in ids
    assert 'b' in ids


def test_clear_trace_overlays_removes_all() -> None:
    """``clear_trace_overlays`` should drop every overlay."""
    viewer = PlotlyRasterViewer()
    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='a', x=[0.0], y=[0.0]))
    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='b', x=[0.0], y=[0.0]))

    viewer.clear_trace_overlays()

    assert list(viewer._plotly_trace_overlays.overlays) == []


def test_sync_trace_overlays_handles_non_list_data() -> None:
    """``_sync_trace_overlays_to_plotly_dict`` should tolerate non-list ``data``."""
    viewer = PlotlyRasterViewer()
    viewer._plotly_dict['data'] = 'not a list'

    viewer.add_trace_overlay(PlotlyTraceOverlay(trace_id='x', x=[0.0], y=[0.0]))

    data = viewer._plotly_dict['data']
    assert isinstance(data, list)


# ---------- request_from_plotly / apply_response / async error paths ----------


def test_request_from_plotly_raises_without_data() -> None:
    """``request_from_plotly`` should raise before ``set_data``."""
    viewer = PlotlyRasterViewer()
    payload = PlotlyViewportPayload(relayout={}, width_px=10, height_px=10)

    with pytest.raises(RuntimeError, match='No data set'):
        viewer.request_from_plotly(payload)


def test_request_from_plotly_returns_view_request_after_set_data() -> None:
    """After ``set_data``, ``request_from_plotly`` should return a ViewRequest."""
    viewer = _viewer_with_data(shape=(8, 4))
    payload = PlotlyViewportPayload(
        relayout={'xaxis.range': [0.0, 16.0], 'yaxis.range': [0.0, 16.0]},
        width_px=200,
        height_px=100,
    )

    req = viewer.request_from_plotly(payload)

    assert req.viewport.width_px == 200
    assert req.viewport.height_px == 100


def test_apply_response_raises_when_plot_not_built() -> None:
    """``apply_response`` should raise if the viewer was never built."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    viewer = PlotlyRasterViewer()
    response = RenderResponse(
        mode='image_png',
        level=0,
        bounds=RowColBounds(row_min=0, row_max=1, col_min=0, col_max=1),
        shape=(1, 1),
        grid=_grid(),
        x0=0.0,
        y0=0.0,
        dx=1.0,
        dy=1.0,
        png_data_uri='data:image/png;base64,AAA',
    )

    with pytest.raises(RuntimeError, match='Viewer must be built'):
        asyncio.run(viewer.apply_response(response))


def test_set_axis_ranges_raises_when_not_built() -> None:
    """``set_axis_ranges`` should raise without a built plot."""
    viewer = PlotlyRasterViewer()

    with pytest.raises(RuntimeError, match='must be built'):
        asyncio.run(viewer.set_axis_ranges(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0))


def test_set_x_axis_range_raises_when_not_built() -> None:
    """``set_x_axis_range`` should raise without a built plot."""
    viewer = PlotlyRasterViewer()

    with pytest.raises(RuntimeError, match='must be built'):
        asyncio.run(viewer.set_x_axis_range(x_min=0.0, x_max=1.0))


def test_set_heatmap_contrast_raises_when_no_trace_active() -> None:
    """``set_heatmap_contrast`` should raise when there is no raster trace."""
    viewer = PlotlyRasterViewer()

    with pytest.raises(RuntimeError, match='No raster trace'):
        asyncio.run(viewer.set_heatmap_contrast(zmin=0.0, zmax=1.0))


def test_set_heatmap_colorscale_raises_when_no_trace_active() -> None:
    """``set_heatmap_colorscale`` should raise when there is no raster trace."""
    viewer = PlotlyRasterViewer()

    with pytest.raises(RuntimeError, match='No raster trace'):
        asyncio.run(viewer.set_heatmap_colorscale('Viridis'))


# ---------- async event handlers (early-return paths) ----------


def test_on_plotly_doubleclick_no_op_without_service() -> None:
    """Doubleclick should be a no-op when no data is loaded."""
    viewer = PlotlyRasterViewer()

    asyncio.run(viewer._on_plotly_doubleclick(types.SimpleNamespace(args={})))


def test_on_plotly_relayout_no_op_without_service() -> None:
    """Relayout should early-return when no data is loaded."""
    viewer = PlotlyRasterViewer()

    asyncio.run(viewer._on_plotly_relayout(types.SimpleNamespace(args={'xaxis.range': [0, 1]})))


def test_on_plotly_relayout_ignores_unrelated_args() -> None:
    """Relayout should early-return when args don't contain axis range keys."""
    viewer = _viewer_with_data(shape=(4, 4))
    viewer._plot = types.SimpleNamespace(id='p')

    asyncio.run(viewer._on_plotly_relayout(types.SimpleNamespace(args={'dragmode': 'pan'})))



def test_on_plotly_relayout_debounces_x_range_emit_and_raster_refresh() -> None:
    """Bracket-key relayout should debounce x-range emit and raster refresh."""
    events: list[tuple[float | None, float | None]] = []

    async def run() -> None:
        viewer = PlotlyRasterViewer(on_x_range_changed=lambda x0, x1: events.append((x0, x1)))
        await viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid())
        viewer._plot = types.SimpleNamespace(id='p')
        refreshed: list[tuple[tuple[float, float], tuple[float, float]]] = []

        async def fake_read_live_viewport_from_browser() -> tuple[
            PlotlyViewportPayload,
            tuple[tuple[float, float], tuple[float, float]],
        ] | None:
            return (
                PlotlyViewportPayload(
                    relayout={
                        'xaxis.range[0]': 2.0,
                        'xaxis.range[1]': 6.0,
                        'yaxis.range[0]': 0.0,
                        'yaxis.range[1]': 8.0,
                    },
                    width_px=100,
                    height_px=50,
                ),
                ((2.0, 6.0), (0.0, 8.0)),
            )

        async def fake_refresh(
            _payload: PlotlyViewportPayload,
            display_axis_ranges: tuple[tuple[float, float], tuple[float, float]],
        ) -> None:
            refreshed.append(display_axis_ranges)

        viewer._read_live_viewport_from_browser = fake_read_live_viewport_from_browser  # type: ignore[method-assign]
        viewer._refresh_raster_for_viewport = fake_refresh  # type: ignore[method-assign]

        await viewer._on_plotly_relayout(
            types.SimpleNamespace(
                args={
                    'xaxis.range[0]': 2.0,
                    'xaxis.range[1]': 6.0,
                    'yaxis.range[0]': 0.0,
                    'yaxis.range[1]': 8.0,
                }
            )
        )

        assert events == []
        assert refreshed == []
        assert viewer._viewport_settle_task is not None
        await viewer._viewport_settle_task

        assert events == [(2.0, 6.0)]
        assert refreshed == [((2.0, 6.0), (0.0, 8.0))]

    asyncio.run(run())


def test_debounced_viewport_settle_coalesces_burst(monkeypatch: pytest.MonkeyPatch) -> None:
    """A burst of relayouts should settle once with the live browser viewport."""
    monkeypatch.setattr(plotly_viewer_module, '_VIEWPORT_SETTLE_DEBOUNCE_SECONDS', 0.0)
    read_count = 0

    async def run() -> None:
        nonlocal read_count
        viewer = PlotlyRasterViewer()
        await viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid())
        viewer._plot = types.SimpleNamespace(id='p')

        async def fake_read_live_viewport_from_browser() -> tuple[
            PlotlyViewportPayload,
            tuple[tuple[float, float], tuple[float, float]],
        ] | None:
            nonlocal read_count
            read_count += 1
            return (
                PlotlyViewportPayload(
                    relayout={'xaxis.range[0]': 2.0, 'xaxis.range[1]': 3.0},
                    width_px=100,
                    height_px=50,
                ),
                ((2.0, 3.0), (0.0, 1.0)),
            )

        viewer._read_live_viewport_from_browser = fake_read_live_viewport_from_browser  # type: ignore[method-assign]
        viewer._refresh_raster_for_viewport = lambda *_args, **_kwargs: asyncio.sleep(0)  # type: ignore[method-assign]

        viewer._schedule_viewport_settle()
        viewer._schedule_viewport_settle()
        assert viewer._viewport_settle_task is not None
        await viewer._viewport_settle_task

    asyncio.run(run())

    assert read_count == 1


def test_debounced_viewport_settle_runs_followup_when_requested_during_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A relayout during settle should trigger one follow-up settle pass."""
    monkeypatch.setattr(plotly_viewer_module, '_VIEWPORT_SETTLE_DEBOUNCE_SECONDS', 0.0)
    read_count = 0

    async def run() -> None:
        nonlocal read_count
        viewer = PlotlyRasterViewer()
        await viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid())
        viewer._plot = types.SimpleNamespace(id='p')

        async def fake_read_live_viewport_from_browser() -> tuple[
            PlotlyViewportPayload,
            tuple[tuple[float, float], tuple[float, float]],
        ] | None:
            nonlocal read_count
            read_count += 1
            return (
                PlotlyViewportPayload(
                    relayout={'xaxis.range[0]': float(read_count), 'xaxis.range[1]': 1.0},
                    width_px=100,
                    height_px=50,
                ),
                ((float(read_count), 1.0), (0.0, 1.0)),
            )

        async def fake_refresh(
            _payload: PlotlyViewportPayload,
            _display: tuple[tuple[float, float], tuple[float, float]],
        ) -> None:
            if read_count == 1:
                viewer._schedule_viewport_settle()

        viewer._read_live_viewport_from_browser = fake_read_live_viewport_from_browser  # type: ignore[method-assign]
        viewer._refresh_raster_for_viewport = fake_refresh  # type: ignore[method-assign]

        viewer._schedule_viewport_settle()
        assert viewer._viewport_settle_task is not None
        await viewer._viewport_settle_task

    asyncio.run(run())

    assert read_count == 2


def test_on_plotly_relayout_roi_shape_relayout_does_not_schedule_settle() -> None:
    """ROI edit relayouts should stay immediate and bypass viewport settle."""
    async def run() -> None:
        viewer = PlotlyRasterViewer()
        await viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid())
        viewer._plot = types.SimpleNamespace(id='p')
        viewer._handle_roi_shape_relayout = lambda _args: True  # type: ignore[method-assign]

        await viewer._on_plotly_relayout(types.SimpleNamespace(args={'shapes[0].x0': 1.0}))

        assert viewer._viewport_settle_requested is False
        assert viewer._viewport_settle_task is None

    asyncio.run(run())


def test_on_plotly_relayout_non_axis_relayout_does_not_schedule_settle() -> None:
    """Non-axis relayout payloads should not schedule viewport settle."""
    async def run() -> None:
        viewer = PlotlyRasterViewer()
        await viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid())
        viewer._plot = types.SimpleNamespace(id='p')

        await viewer._on_plotly_relayout(types.SimpleNamespace(args={'dragmode': 'pan'}))

        assert viewer._viewport_settle_requested is False
        assert viewer._viewport_settle_task is None

    asyncio.run(run())


def test_display_axis_ranges_from_relayout_uses_cache_for_missing_axis() -> None:
    """A partial axis relayout should preserve the other displayed axis."""
    viewer = _viewer_with_data(shape=(4, 4))
    viewer._last_display_axis_ranges = ((10.0, 20.0), (30.0, 40.0))

    ranges = viewer._display_axis_ranges_from_relayout(
        {'xaxis.range[0]': 11.0, 'xaxis.range[1]': 12.0}
    )

    assert ranges == ((11.0, 12.0), (30.0, 40.0))


def test_on_plotly_relayout_ignores_normalized_echo_of_last_display_viewport() -> None:
    """Post-``set_data`` normalized relayout echoes must not schedule viewport settle."""

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        await viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid())
        viewer._plot = types.SimpleNamespace(id='p')
        assert viewer._last_display_axis_ranges is not None
        (x_lo, x_hi), (y_lo, y_hi) = viewer._last_display_axis_ranges

        await viewer._on_plotly_relayout(
            types.SimpleNamespace(
                args={
                    'xaxis.range': [x_lo, x_hi],
                    'xaxis.autorange': False,
                    'yaxis.range': [y_lo, y_hi],
                    'yaxis.autorange': False,
                }
            )
        )

        assert viewer._viewport_settle_requested is False
        assert viewer._viewport_settle_task is None

    asyncio.run(run())


def test_on_plotly_relayout_schedules_settle_for_normalized_user_zoom() -> None:
    """Scroll-zoom normalized relayout must schedule viewport settle when ranges differ."""

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        await viewer.set_data(np.zeros((4, 4), dtype=np.float32), grid=_grid())
        viewer._plot = types.SimpleNamespace(id='p')

        await viewer._on_plotly_relayout(
            types.SimpleNamespace(
                args={
                    'xaxis.range': [1.0, 2.0],
                    'xaxis.autorange': False,
                    'yaxis.range': [3.0, 4.0],
                    'yaxis.autorange': False,
                }
            )
        )

        assert viewer._viewport_settle_requested is True
        assert viewer._viewport_settle_task is not None

    asyncio.run(run())


def test_apply_response_preserves_display_axis_ranges_via_restyle() -> None:
    """Viewport-driven raster swaps should restyle without NiceGUI figure push."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    class FakeClient:
        def __init__(self) -> None:
            self.js_calls: list[str] = []

        async def run_javascript(self, js: str, timeout: float) -> None:
            self.js_calls.append(js)

    class FakePlot:
        def __init__(self) -> None:
            self.id = 'plot-id'
            self.figure = {}
            self.updated = False
            self.client = FakeClient()

        def update(self) -> None:
            self.updated = True

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        fake_plot = FakePlot()
        viewer._plot = fake_plot
        viewer._plotly_dict = {
            'data': [{'type': 'image', 'source': 'old', 'x0': 0.0, 'y0': 0.0, 'dx': 1.0, 'dy': 1.0}],
            'layout': {'xaxis': {'range': [0.0, 1.0]}, 'yaxis': {'range': [0.0, 1.0]}},
        }
        response = RenderResponse(
            mode='image_png',
            level=0,
            bounds=RowColBounds(row_min=0, row_max=4, col_min=0, col_max=4),
            shape=(4, 4),
            grid=_grid(),
            x0=0.0,
            y0=0.0,
            dx=2.0,
            dy=4.0,
            png_data_uri='data:image/png;base64,AAA',
        )

        await viewer.apply_response(response, display_axis_ranges=((3.0, 7.0), (5.0, 9.0)))

        assert fake_plot.updated is False
        assert len(fake_plot.client.js_calls) == 1
        assert 'Plotly.restyle' in fake_plot.client.js_calls[0]
        assert viewer.figure['layout']['xaxis']['range'] == [3.0, 7.0]
        assert viewer.figure['layout']['yaxis']['range'] == [5.0, 9.0]
        assert viewer._last_display_axis_ranges == ((3.0, 7.0), (5.0, 9.0))

    asyncio.run(run())


def test_read_live_viewport_from_browser_uses_plot_client_and_caches_size() -> None:
    """Live viewport read should use the explicit Plotly element client."""

    class FakeClient:
        async def run_javascript(self, js: str, timeout: float) -> dict[str, object]:
            assert 'layout.xaxis' in js
            assert timeout == 2.0
            return {
                'x_range': [1.0, 2.0],
                'y_range': [3.0, 4.0],
                'width_px': 321,
                'height_px': 123,
            }

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        viewer._plot = types.SimpleNamespace(id='p', client=FakeClient())

        settled = await viewer._read_live_viewport_from_browser()

        assert settled is not None
        payload, display = settled
        assert payload.width_px == 321
        assert payload.height_px == 123
        assert display == ((1.0, 2.0), (3.0, 4.0))
        assert viewer._last_viewport_size_px == (321, 123)

    asyncio.run(run())


def test_read_live_viewport_from_browser_returns_none_when_js_fails() -> None:
    """Viewport settle should skip when the browser state is unavailable."""

    class FakeClient:
        async def run_javascript(self, js: str, timeout: float) -> None:
            raise RuntimeError('client context unavailable')

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        viewer._plot = types.SimpleNamespace(id='p', client=FakeClient())

        settled = await viewer._read_live_viewport_from_browser()

        assert settled is None

    asyncio.run(run())


def test_set_x_axis_range_noop_when_display_x_unchanged() -> None:
    """``set_x_axis_range`` should skip JS when the display x-range already matches."""

    class FakeClient:
        def __init__(self) -> None:
            self.js_calls = 0

        def run_javascript(self, js: str, timeout: float) -> None:
            self.js_calls += 1

    class FakePlot:
        id = 'plot-id'
        client = FakeClient()

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        viewer._plot = FakePlot()
        viewer._transform = PlotlyCoordTransform(nrows=4, ncols=4, grid=_grid())
        viewer._last_display_axis_ranges = ((1.0, 3.0), (0.0, 8.0))
        await viewer.set_x_axis_range(x_min=1.0, x_max=3.0)
        assert viewer._plot.client.js_calls == 0

    asyncio.run(run())


def test_set_x_axis_range_relayouts_x_axis_only() -> None:
    """``set_x_axis_range`` must not send ``yaxis.range`` relayout keys."""

    class FakeClient:
        def __init__(self) -> None:
            self.last_js = ''

        def run_javascript(self, js: str, timeout: float) -> None:
            self.last_js = js

    class FakePlot:
        id = 'plot-id'
        client = FakeClient()

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        viewer._plot = FakePlot()
        viewer._transform = PlotlyCoordTransform(nrows=4, ncols=4, grid=_grid())
        viewer._last_display_axis_ranges = ((0.0, 4.0), (1.0, 2.0))
        await viewer.set_x_axis_range(x_min=1.0, x_max=3.0)
        assert 'xaxis.range' in viewer._plot.client.last_js
        assert 'yaxis.range' not in viewer._plot.client.last_js

    asyncio.run(run())


def test_reset_x_axis_to_full_extent_relayouts_x_axis_only() -> None:
    """``reset_x_axis_to_full_extent`` must not send ``yaxis.range`` relayout keys."""

    class FakeClient:
        def __init__(self) -> None:
            self.last_js = ''

        def run_javascript(self, js: str, timeout: float) -> None:
            self.last_js = js

    class FakePlot:
        id = 'plot-id'
        client = FakeClient()

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        viewer._plot = FakePlot()
        viewer._transform = PlotlyCoordTransform(nrows=4, ncols=4, grid=_grid())
        viewer._plotly_dict = {'layout': {'xaxis': {}, 'yaxis': {}}}
        viewer._last_display_axis_ranges = ((1.0, 2.0), (5.0, 6.0))
        viewer._schedule_viewport_settle = lambda: None  # type: ignore[method-assign]
        await viewer.reset_x_axis_to_full_extent()
        assert 'xaxis.range' in viewer._plot.client.last_js
        assert 'yaxis.range' not in viewer._plot.client.last_js
        assert viewer._last_display_axis_ranges == ((0.0, 8.0), (5.0, 6.0))

    asyncio.run(run())


def test_apply_response_reacts_on_trace_type_change_without_nicegui_update() -> None:
    """PNG/heatmap switches should use Plotly.react and preserve browser layout."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    class FakeClient:
        def __init__(self) -> None:
            self.js_calls: list[str] = []

        async def run_javascript(self, js: str, timeout: float) -> None:
            self.js_calls.append(js)

    class FakePlot:
        def __init__(self) -> None:
            self.id = 'plot-id'
            self.figure = {}
            self.updated = False
            self.client = FakeClient()

        def update(self) -> None:
            self.updated = True

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        fake_plot = FakePlot()
        viewer._plot = fake_plot
        viewer._plotly_dict = {
            'data': [{'type': 'image', 'source': 'old', 'x0': 0.0, 'y0': 0.0, 'dx': 1.0, 'dy': 1.0}],
            'layout': {'xaxis': {'range': [0.0, 1.0]}, 'yaxis': {'range': [0.0, 1.0]}},
            'config': {},
        }
        z = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        response = RenderResponse(
            mode='heatmap_z',
            level=0,
            bounds=RowColBounds(row_min=0, row_max=2, col_min=0, col_max=2),
            shape=(2, 2),
            grid=_grid(),
            x0=0.0,
            y0=0.0,
            dx=2.0,
            dy=4.0,
            z=z,
            zmin=1.0,
            zmax=4.0,
        )

        await viewer.apply_response(response, display_axis_ranges=((0.5, 1.5), (2.0, 6.0)))

        assert fake_plot.updated is False
        assert len(fake_plot.client.js_calls) == 1
        assert 'Plotly.react' in fake_plot.client.js_calls[0]
        assert 'plotDiv.layout' in fake_plot.client.js_calls[0]

    asyncio.run(run())


def test_apply_response_restyles_same_image_trace_without_layout_update() -> None:
    """Relayout refresh with same image trace should restyle trace 0 only."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    class FakeClient:
        def __init__(self) -> None:
            self.js_calls: list[str] = []

        async def run_javascript(self, js: str, timeout: float) -> None:
            self.js_calls.append(js)
            assert timeout == 10.0

    class FakePlot:
        def __init__(self) -> None:
            self.id = 'plot-id'
            self.figure = {}
            self.updated = False
            self.client = FakeClient()

        def update(self) -> None:
            self.updated = True

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        fake_plot = FakePlot()
        viewer._plot = fake_plot
        viewer._plotly_dict = {
            'data': [{'type': 'image', 'source': 'old', 'x0': 0.0, 'y0': 0.0, 'dx': 1.0, 'dy': 1.0}],
            'layout': {'xaxis': {'range': [10.0, 20.0]}, 'yaxis': {'range': [30.0, 40.0]}},
        }
        response = RenderResponse(
            mode='image_png',
            level=1,
            bounds=RowColBounds(row_min=1, row_max=3, col_min=2, col_max=4),
            shape=(10, 10),
            grid=_grid(),
            x0=4.0,
            y0=8.0,
            dx=2.0,
            dy=4.0,
            png_data_uri='data:image/png;base64,NEW',
        )

        await viewer.apply_response(response, display_axis_ranges=((10.0, 20.0), (30.0, 40.0)))

        assert fake_plot.updated is False
        assert len(fake_plot.client.js_calls) == 1
        js = fake_plot.client.js_calls[0]
        assert 'Plotly.restyle' in js
        assert 'Plotly.relayout' not in js
        assert 'xaxis' not in js
        assert 'yaxis' not in js
        assert viewer.figure['layout']['xaxis']['range'] == [10.0, 20.0]
        assert viewer.figure['layout']['yaxis']['range'] == [30.0, 40.0]
        assert viewer.figure['data'][0]['source'] == 'data:image/png;base64,NEW'

    asyncio.run(run())


def test_apply_response_restyles_same_heatmap_trace_without_layout_update() -> None:
    """Relayout refresh with same heatmap trace should restyle trace 0 only."""
    from nicewidgets.raster_viewer.backend.image_model import RenderResponse

    class FakeClient:
        def __init__(self) -> None:
            self.js_calls: list[str] = []

        async def run_javascript(self, js: str, timeout: float) -> None:
            self.js_calls.append(js)
            assert timeout == 10.0

    class FakePlot:
        def __init__(self) -> None:
            self.id = 'plot-id'
            self.figure = {}
            self.updated = False
            self.client = FakeClient()

        def update(self) -> None:
            self.updated = True

    async def run() -> None:
        viewer = PlotlyRasterViewer()
        fake_plot = FakePlot()
        viewer._plot = fake_plot
        viewer._plotly_dict = {
            'data': [{'type': 'heatmap', 'z': [[0.0]], 'x0': 0.0, 'y0': 0.0, 'dx': 1.0, 'dy': 1.0}],
            'layout': {'xaxis': {'range': [1.0, 2.0]}, 'yaxis': {'range': [3.0, 4.0]}},
        }
        response = RenderResponse(
            mode='heatmap_z',
            level=0,
            bounds=RowColBounds(row_min=0, row_max=2, col_min=0, col_max=2),
            shape=(2, 2),
            grid=_grid(),
            x0=0.0,
            y0=0.0,
            dx=2.0,
            dy=4.0,
            z=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            zmin=1.0,
            zmax=4.0,
        )

        await viewer.apply_response(response, display_axis_ranges=((1.0, 2.0), (3.0, 4.0)))

        assert fake_plot.updated is False
        assert len(fake_plot.client.js_calls) == 1
        js = fake_plot.client.js_calls[0]
        assert 'Plotly.restyle' in js
        assert 'Plotly.relayout' not in js
        assert 'xaxis' not in js
        assert 'yaxis' not in js
        assert viewer.figure['layout']['xaxis']['range'] == [1.0, 2.0]
        assert viewer.figure['layout']['yaxis']['range'] == [3.0, 4.0]
        assert viewer.figure['data'][0]['z'] == [[1.0, 2.0], [3.0, 4.0]]

    asyncio.run(run())


def test_get_viewport_returns_last_display_ranges_after_set_data() -> None:
    """``get_viewport`` should expose the viewer's cached Plotly axis ranges."""
    viewer = PlotlyRasterViewer()
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='s', y_unit='um')

    asyncio.run(viewer.set_data(data, grid=grid))

    viewport = viewer.get_viewport()
    assert viewport is not None
    assert viewport[0] == (0.0, 10.0)
    assert viewport[1] == (0.0, 10.0)


def test_swap_slice_plane_preserves_zoomed_viewport() -> None:
    """Slice reloads should render at the cached viewport without a full reset."""
    from nicewidgets.raster_viewer.backend.image_model import BackendImage
    from nicewidgets.raster_viewer.backend.pyramid import ImagePyramid
    from nicewidgets.raster_viewer.frontend.plotly_viewer import DisplayAxisRanges

    viewer = PlotlyRasterViewer()
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='s', y_unit='um')
    asyncio.run(viewer.set_data(data, grid=grid))

    zoomed: DisplayAxisRanges = ((2.0, 6.0), (3.0, 7.0))
    viewer._last_display_axis_ranges = zoomed  # noqa: SLF001
    viewer._last_viewport_size_px = (400, 300)  # noqa: SLF001

    next_plane = np.arange(100, 200, dtype=np.float32).reshape(10, 10)
    pyramid = ImagePyramid(BackendImage(next_plane, grid=grid))

    apply_calls: list[DisplayAxisRanges | None] = []

    async def _fake_apply(response, *, display_axis_ranges=None) -> None:  # type: ignore[no-untyped-def]
        apply_calls.append(display_axis_ranges)

    viewer.apply_response = _fake_apply  # type: ignore[method-assign]

    asyncio.run(
        viewer.swap_slice_plane(
            next_plane,
            grid=grid,
            pyramid=pyramid,
            display_axis_ranges=zoomed,
        )
    )

    assert apply_calls == [zoomed]
    assert viewer.get_viewport() == zoomed
