"""Tests for PlotlyRasterViewer trace overlay state."""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np

if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.ui = types.SimpleNamespace()
    sys.modules['nicegui'] = fake_nicegui

from nicewidgets.plotly_layout_margins import PlotlyLayoutMarginsProfile
from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec
from nicewidgets.raster_viewer.frontend.plotly_display_options import (
    PlotlyRasterViewerDisplayOptions,
)
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer
from nicewidgets.raster_viewer.frontend.trace_overlay import PlotlyTraceOverlay


def test_set_data_clears_trace_overlays() -> None:
    """Loading a new dataset should clear stale trace overlays."""
    viewer = PlotlyRasterViewer()
    viewer.add_trace_overlay(
        PlotlyTraceOverlay(trace_id='left', x=[0.0, 1.0], y=[2.0, 3.0])
    )

    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='s', y_unit='um')

    asyncio.run(viewer.set_data(data, grid=grid))

    figure_data = viewer.figure.get('data')
    assert isinstance(figure_data, list)
    assert len(figure_data) == 1


def test_set_data_from_pyramid_reuses_prebuilt_pyramid() -> None:
    """Prebuilt pyramids should bypass a second pyramid build in the viewer."""
    from nicewidgets.raster_viewer.backend.image_model import BackendImage
    from nicewidgets.raster_viewer.backend.pyramid import ImagePyramid

    viewer = PlotlyRasterViewer()
    data = np.arange(36, dtype=np.float32).reshape(6, 6)
    grid = RasterGridSpec(dx=0.5, dy=0.25, x_unit='s', y_unit='um')
    pyramid = ImagePyramid(BackendImage(data, grid=grid))

    asyncio.run(viewer.set_data_from_pyramid(data, grid=grid, pyramid=pyramid))

    assert viewer.has_data
    assert viewer.figure.get('data') is not None


def test_raster_display_options_round_trip_excludes_layout_margins() -> None:
    """Serialization should preserve scalar fields and drop layout margins."""
    options = PlotlyRasterViewerDisplayOptions(
        show_plotly_toolbar=True,
        show_rois=False,
        show_roi_labels=False,
        show_x_axis_labels=True,
        square_plot=True,
        theme='dark',
        layout_margins_profile=PlotlyLayoutMarginsProfile(
            with_axis_labels={'l': 40, 'r': 10, 't': 10, 'b': 40},
            compact={'l': 5, 'r': 5, 't': 5, 'b': 5},
        ),
    )

    data = options.to_dict()
    assert 'layout_margins_profile' not in data
    assert data['theme'] == 'dark'

    restored = PlotlyRasterViewerDisplayOptions.from_dict(data)
    assert restored.show_plotly_toolbar is True
    assert restored.show_rois is False
    assert restored.show_roi_labels is False
    assert restored.show_x_axis_labels is True
    assert restored.square_plot is True
    assert restored.theme == 'dark'
    assert restored.layout_margins_profile is None


def test_raster_display_options_from_dict_ignores_unknown_keys() -> None:
    """Unknown/legacy keys must be ignored and missing keys use defaults."""
    restored = PlotlyRasterViewerDisplayOptions.from_dict(
        {'show_rois': False, 'layout_margins_profile': {'stale': True}}
    )
    assert restored.show_rois is False
    assert restored.show_plotly_toolbar is False
    assert restored.layout_margins_profile is None