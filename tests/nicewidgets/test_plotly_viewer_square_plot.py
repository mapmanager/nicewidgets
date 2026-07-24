"""Headless tests for PlotlyRasterViewer square layout, padding, and ROI edit mode."""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np
import pytest

if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.ui = types.SimpleNamespace()
    fake_nicegui.run = types.SimpleNamespace()
    fake_nicegui.app = types.SimpleNamespace(native=types.SimpleNamespace(main_window=None))
    sys.modules['nicegui'] = fake_nicegui

from nicewidgets.raster_viewer.backend.image_model import (
    BackendImage,
    RasterDisplayStyle,
    RasterGridSpec,
    RenderResponse,
    RowColBounds,
)
from nicewidgets.raster_viewer.backend.pyramid import ImagePyramid
from nicewidgets.raster_viewer.frontend.plotly_coord_transform import PlotlyCoordTransform
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer
from nicewidgets.raster_viewer.frontend.roi_overlay import RectRoiOverlay

_GRID = RasterGridSpec(dx=1.0, dy=1.0, x_unit='', y_unit='')


def test_clear_data_resets_viewer_to_empty_figure(monkeypatch: pytest.MonkeyPatch) -> None:
    """clear_data should drop backend state and restore an empty figure."""
    captured: dict[str, object] = {}

    class DummyElement:
        id = 1
        figure: dict[str, object] | None = None

        def on(self, *_args, **_kwargs) -> DummyElement:
            return self

        def update(self) -> None:
            return None

    class DummyContextMenu:
        def clear(self) -> DummyContextMenu:
            return self

        def __enter__(self) -> DummyContextMenu:
            return self

        def __exit__(self, *_args) -> None:
            return None

        def open(self) -> None:
            return None

    class DummyUI:
        @staticmethod
        def plotly(figure):
            captured['figure'] = figure
            return DummyElement()

        @staticmethod
        def context_menu() -> DummyContextMenu:
            return DummyContextMenu()

    from nicewidgets.raster_viewer.frontend import plotly_viewer as plotly_viewer_module

    monkeypatch.setattr(
        plotly_viewer_module,
        'ui',
        types.SimpleNamespace(plotly=DummyUI.plotly, context_menu=DummyUI.context_menu),
    )

    viewer = PlotlyRasterViewer()
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    asyncio.run(viewer.set_data(data, grid=_GRID))
    assert viewer.has_data is True

    asyncio.run(viewer.clear_data())

    assert viewer.has_data is False
    assert viewer.figure['data'] == []


def test_set_data_auto_enables_square_plot_for_square_source() -> None:
    """Square source arrays should initialize with square Plotly constraints."""
    viewer = PlotlyRasterViewer()
    data = np.arange(16, dtype=np.float32).reshape(4, 4)

    asyncio.run(viewer.set_data(data, grid=_GRID))

    layout = viewer.figure['layout']
    assert viewer.display_options.square_plot is True
    assert layout['xaxis']['constrain'] == 'domain'
    assert layout['yaxis']['constrain'] == 'domain'
    assert layout['yaxis']['scaleanchor'] == 'x'
    assert layout['yaxis']['scaleratio'] == 1.0


def test_set_data_auto_disables_square_plot_for_non_square_source() -> None:
    """Non-square source arrays should initialize without square constraints."""
    viewer = PlotlyRasterViewer()
    data = np.arange(32, dtype=np.float32).reshape(4, 8)

    asyncio.run(viewer.set_data(data, grid=_GRID))

    layout = viewer.figure['layout']
    assert viewer.display_options.square_plot is False
    assert 'constrain' not in layout['xaxis']
    assert 'constrain' not in layout['yaxis']
    assert layout['yaxis']['scaleanchor'] is False
    assert 'scaleratio' not in layout['yaxis']


def test_set_square_plot_can_force_non_square_source_square() -> None:
    """The context-menu action can force square layout for any current source."""
    viewer = PlotlyRasterViewer()
    data = np.arange(32, dtype=np.float32).reshape(4, 8)
    asyncio.run(viewer.set_data(data, grid=_GRID))

    viewer.set_square_plot(True)

    layout = viewer.figure['layout']
    assert viewer.display_options.square_plot is True
    assert layout['xaxis']['constrain'] == 'domain'
    assert layout['yaxis']['constrain'] == 'domain'
    assert layout['yaxis']['scaleanchor'] == 'x'
    assert layout['yaxis']['scaleratio'] == 0.5


def test_set_data_reapplies_auto_square_plot_after_user_toggle() -> None:
    """Loading new data should re-auto-evaluate square layout state."""
    viewer = PlotlyRasterViewer()
    non_square = np.arange(32, dtype=np.float32).reshape(4, 8)
    asyncio.run(viewer.set_data(non_square, grid=_GRID))
    viewer.set_square_plot(True)

    asyncio.run(viewer.set_data(non_square, grid=_GRID))

    layout = viewer.figure['layout']
    assert viewer.display_options.square_plot is False
    assert layout['yaxis']['scaleanchor'] is False
    assert 'scaleratio' not in layout['yaxis']


def test_set_x_axis_range_preserves_y_row_col_extent() -> None:
    """set_x_axis_range should update row span only; column span unchanged."""
    class _DummyClient:
        def run_javascript(self, *_args, **_kwargs) -> None:
            return None

    class _DummyPlot:
        id = 99
        client = _DummyClient()

    viewer = PlotlyRasterViewer()
    viewer._plot = _DummyPlot()
    viewer._transform = PlotlyCoordTransform(nrows=4, ncols=8, grid=_GRID)
    viewer._current_bounds = RowColBounds(
        row_min=0.0,
        row_max=4.0,
        col_min=0.0,
        col_max=8.0,
    )

    asyncio.run(viewer.set_x_axis_range(x_min=1.0, x_max=3.0))

    b = viewer.current_bounds
    assert (b.row_min, b.row_max) == (1.0, 3.0)
    assert (b.col_min, b.col_max) == (0.0, 8.0)


def test_set_roi_editing_marks_only_active_shape_editable() -> None:
    """ROI edit mode should make only the target shape editable."""
    viewer = PlotlyRasterViewer()
    viewer.set_rois(
        [
            RectRoiOverlay(roi_id=1, x0=0.0, x1=1.0, y0=0.0, y1=1.0),
            RectRoiOverlay(roi_id=2, x0=2.0, x1=3.0, y0=2.0, y1=3.0),
        ]
    )

    viewer.set_roi_editing(True, 2)

    shapes = viewer.figure['layout']['shapes']
    editable_by_name = {shape['name']: shape['editable'] for shape in shapes}
    assert editable_by_name == {'roi:1': False, 'roi:2': True}
    assert viewer.figure['config']['edits']['shapePosition'] is True

    viewer.set_roi_editing(False, 2)

    idle_shapes = viewer.figure['layout']['shapes']
    assert {shape['name']: shape['editable'] for shape in idle_shapes} == {'roi:1': False, 'roi:2': False}
    assert viewer.figure['config']['edits']['shapePosition'] is False


def test_set_data_updates_plot_with_unpadded_full_extent() -> None:
    """Initial render should use the real unpadded data extent."""

    class _FakePlot:
        id = 'plot-id'

        def __init__(self) -> None:
            self.figure: dict[str, object] = {}
            self.ranges_at_update: tuple[list[float], list[float]] | None = None

        def update(self) -> None:
            layout = self.figure['layout']
            self.ranges_at_update = (
                list(layout['xaxis']['range']),
                list(layout['yaxis']['range']),
            )

    async def _run() -> None:
        viewer = PlotlyRasterViewer()
        fake_plot = _FakePlot()
        viewer._plot = fake_plot
        data = np.arange(32, dtype=np.float32).reshape(4, 8)
        pyramid = ImagePyramid(BackendImage(data, grid=_GRID))

        await viewer.set_data_from_pyramid(data, grid=_GRID, pyramid=pyramid)

        assert fake_plot.ranges_at_update is not None
        x_range, y_range = fake_plot.ranges_at_update
        assert x_range == pytest.approx([0.0, 4.0])
        assert y_range == pytest.approx([0.0, 8.0])
        assert viewer._last_display_axis_ranges is not None
        assert viewer._last_display_axis_ranges[0] == pytest.approx((0.0, 4.0))
        assert viewer._last_display_axis_ranges[1] == pytest.approx((0.0, 8.0))

    asyncio.run(_run())


def test_refresh_full_png_applies_full_response() -> None:
    """Contrast PNG refreshes should use the baseline full-response path."""

    response = RenderResponse(
        mode='image_png',
        level=0,
        bounds=RowColBounds(row_min=0.0, row_max=4.0, col_min=0.0, col_max=8.0),
        shape=(4, 8),
        grid=_GRID,
        x0=0.0,
        y0=0.0,
        dx=1.0,
        dy=1.0,
        png_data_uri='data:image/png;base64,',
    )

    class _FakeService:
        def full_image_png(
            self,
            *,
            display_style: RasterDisplayStyle,
            max_pixels: int | None,
        ) -> RenderResponse:
            return response

    async def _run() -> None:
        viewer = PlotlyRasterViewer()
        viewer._plot = object()
        viewer._service = _FakeService()
        captured: dict[str, object] = {}

        async def _capture_apply_response(
            response_arg: RenderResponse,
            *,
            display_axis_ranges=None,
        ) -> None:
            captured['response'] = response_arg
            captured['display_axis_ranges'] = display_axis_ranges

        viewer.apply_response = _capture_apply_response  # type: ignore[method-assign]

        await viewer._refresh_full_png()

        assert captured == {'response': response, 'display_axis_ranges': None}

    asyncio.run(_run())


def test_doubleclick_reset_applies_full_unpadded_extent() -> None:
    """Double-click full reset should apply the current image's real full extent."""

    response = RenderResponse(
        mode='image_png',
        level=0,
        bounds=RowColBounds(row_min=0.0, row_max=4.0, col_min=0.0, col_max=8.0),
        shape=(4, 8),
        grid=_GRID,
        x0=0.0,
        y0=0.0,
        dx=1.0,
        dy=1.0,
        png_data_uri='data:image/png;base64,NEW',
    )

    class _FakeService:
        def full_image_png(
            self,
            *,
            display_style: RasterDisplayStyle,
            max_pixels: int | None,
        ) -> RenderResponse:
            return response

    class _FakeClient:
        def __init__(self) -> None:
            self.js_calls: list[str] = []

        async def run_javascript(self, js: str, timeout: float) -> None:
            self.js_calls.append(js)
            assert timeout == 10.0

    class _FakePlot:
        def __init__(self) -> None:
            self.id = 'plot-id'
            self.figure: dict[str, object] = {}
            self.updated = False
            self.client = _FakeClient()

        def update(self) -> None:
            self.updated = True

    async def _run() -> None:
        x_range_events: list[tuple[float | None, float | None]] = []
        viewer = PlotlyRasterViewer(on_x_range_changed=lambda *args: x_range_events.append(args))
        fake_plot = _FakePlot()
        viewer._plot = fake_plot
        viewer._service = _FakeService()
        viewer._transform = PlotlyCoordTransform(nrows=4, ncols=8, grid=_GRID)
        viewer._plotly_dict = {
            'data': [{'type': 'image', 'source': 'old', 'x0': 0.0, 'y0': 0.0, 'dx': 1.0, 'dy': 1.0}],
            'layout': {'uirevision': 'old-ui', 'xaxis': {'range': [1.0, 2.0]}, 'yaxis': {'range': [3.0, 4.0]}},
            'config': {},
        }
        viewer._new_uirevision = lambda: 'reset-ui'  # type: ignore[method-assign]

        await viewer._on_plotly_doubleclick(object())

        assert fake_plot.updated is True
        assert fake_plot.client.js_calls == []
        assert viewer.figure['layout']['uirevision'] == 'reset-ui'
        assert viewer.figure['layout']['xaxis']['range'] == pytest.approx([0.0, 4.0])
        assert viewer.figure['layout']['yaxis']['range'] == pytest.approx([0.0, 8.0])
        assert viewer._last_display_axis_ranges is not None
        assert viewer._last_display_axis_ranges[0] == pytest.approx((0.0, 4.0))
        assert viewer._last_display_axis_ranges[1] == pytest.approx((0.0, 8.0))
        assert x_range_events == [(None, None)]

    asyncio.run(_run())


def test_doubleclick_after_second_file_resets_to_second_file_extent() -> None:
    """Reset after a file switch should use the second file's unpadded extent."""

    class _FakePlot:
        id = 'plot-id'

        def __init__(self) -> None:
            self.figure: dict[str, object] = {}
            self.update_count = 0

        def update(self) -> None:
            self.update_count += 1

    async def _run() -> None:
        x_range_events: list[tuple[float | None, float | None]] = []
        viewer = PlotlyRasterViewer(on_x_range_changed=lambda *args: x_range_events.append(args))
        fake_plot = _FakePlot()
        viewer._plot = fake_plot
        revisions = iter(['file-one', 'file-two', 'reset-two'])
        viewer._new_uirevision = lambda: next(revisions)  # type: ignore[method-assign]

        first = np.zeros((4, 8), dtype=np.float32)
        first_pyramid = ImagePyramid(BackendImage(first, grid=_GRID))
        await viewer.set_data_from_pyramid(first, grid=_GRID, pyramid=first_pyramid)

        second = np.zeros((6, 3), dtype=np.float32)
        second_pyramid = ImagePyramid(BackendImage(second, grid=_GRID))
        await viewer.set_data_from_pyramid(second, grid=_GRID, pyramid=second_pyramid)

        viewer.figure['layout']['xaxis']['range'] = [2.0, 3.0]
        viewer.figure['layout']['yaxis']['range'] = [1.0, 2.0]
        viewer._last_display_axis_ranges = ((2.0, 3.0), (1.0, 2.0))

        await viewer._on_plotly_doubleclick(object())

        assert fake_plot.update_count == 3
        assert viewer.figure['layout']['uirevision'] == 'reset-two'
        assert viewer.figure['layout']['xaxis']['range'] == pytest.approx([0.0, 6.0])
        assert viewer.figure['layout']['yaxis']['range'] == pytest.approx([0.0, 3.0])
        assert viewer._last_display_axis_ranges == ((0.0, 6.0), (0.0, 3.0))
        assert x_range_events == [(None, None)]

    asyncio.run(_run())


def test_roi_shape_relayout_updates_overlay_and_emits_preview() -> None:
    """Shape relayout during edit mode should update the active ROI preview."""
    previews: list[tuple[int, float, float, float, float]] = []
    viewer = PlotlyRasterViewer(on_roi_bounds_preview=lambda *args: previews.append(args))
    viewer.set_rois([RectRoiOverlay(roi_id=7, x0=1.0, x1=4.0, y0=2.0, y1=5.0)])
    viewer.set_roi_editing(True, 7)

    handled = viewer._handle_roi_shape_relayout(
        {
            'shapes[0].x0': 2.0,
            'shapes[0].x1': 6.0,
            'shapes[0].y0': 3.0,
            'shapes[0].y1': 8.0,
        }
    )

    assert handled is True
    assert previews == [(7, 2.0, 6.0, 3.0, 8.0)]
    shape = viewer.figure['layout']['shapes'][0]
    assert (shape['x0'], shape['x1'], shape['y0'], shape['y1']) == (2.0, 6.0, 3.0, 8.0)
