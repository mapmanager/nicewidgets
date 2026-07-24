"""Tests for Plotly raster viewer context-menu display state."""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np

if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.ui = types.SimpleNamespace()
    sys.modules['nicegui'] = fake_nicegui

from nicewidgets.plotly_axis_layout import resolve_plot_layout_margins
from nicewidgets.plotly_layout_margins import PlotlyLayoutMarginsProfile
from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec
from nicewidgets.raster_viewer.frontend.plotly_context_menu import (
    PlotlyRasterViewerContextMenu,
)
from nicewidgets.raster_viewer.frontend.plotly_display_options import (
    PlotlyRasterViewerDisplayOptions,
)
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer
from nicewidgets.raster_viewer.frontend.roi_overlay import RectRoiOverlay
from nicewidgets.raster_viewer.frontend.trace_overlay import PlotlyTraceOverlay

_LABELED_MARGINS = resolve_plot_layout_margins(show_axis_labels=True, show_legend=False)
_COMPACT_MARGINS = resolve_plot_layout_margins(show_axis_labels=False, show_legend=False)


def _viewer_with_data(
    *,
    display_options: PlotlyRasterViewerDisplayOptions | None = None,
) -> PlotlyRasterViewer:
    """Return a headless viewer with a small raster dataset loaded."""
    viewer = PlotlyRasterViewer(display_options=display_options)
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=2.0, x_unit='s', y_unit='um')
    asyncio.run(viewer.set_data(data, grid=grid))
    return viewer


def test_display_options_defaults_match_context_menu_requirements() -> None:
    """Display options should default to toolbar off, overlays on, labels off."""
    options = PlotlyRasterViewerDisplayOptions()

    assert options.show_plotly_toolbar is False
    assert options.show_rois is True
    assert options.show_trace_overlays is True
    assert options.show_x_axis_labels is False
    assert options.show_y_axis_labels is False
    assert options.show_hover_info is False
    assert options.theme == 'light'


def test_viewer_accepts_caller_supplied_display_options() -> None:
    """Callers should be able to provide initial display options."""
    options = PlotlyRasterViewerDisplayOptions(
        show_plotly_toolbar=True,
        show_rois=False,
        show_trace_overlays=False,
        show_x_axis_labels=True,
        show_y_axis_labels=True,
        theme='dark',
    )

    viewer = _viewer_with_data(display_options=options)
    layout = viewer.figure['layout']

    assert viewer.display_options is options
    assert viewer.figure['config']['displayModeBar'] is True
    assert layout['xaxis']['title']['text'] == 's'
    assert layout['yaxis']['title']['text'] == 'um'
    assert layout['margin'] == _LABELED_MARGINS
    assert layout['paper_bgcolor'] == '#111827'


def test_axis_labels_are_hidden_by_default_but_preserved_for_toggle() -> None:
    """Axis labels should be blank by default and restored when enabled."""
    viewer = _viewer_with_data()
    layout = viewer.figure['layout']

    assert layout['xaxis']['title']['text'] == ''
    assert layout['yaxis']['title']['text'] == ''
    assert layout['margin'] == _COMPACT_MARGINS

    viewer.set_x_axis_labels_visible(True)
    viewer.set_y_axis_labels_visible(True)

    assert viewer.figure['layout']['xaxis']['title']['text'] == 's'
    assert viewer.figure['layout']['yaxis']['title']['text'] == 'um'
    assert viewer.figure['layout']['margin'] == _LABELED_MARGINS


def test_axis_label_margin_toggle_swaps_between_compact_and_labeled() -> None:
    """Margins should shrink when axis labels are hidden and expand when shown."""
    viewer = _viewer_with_data()

    assert viewer.figure['layout']['margin'] == _COMPACT_MARGINS

    viewer.set_x_axis_labels_visible(True)
    viewer.set_y_axis_labels_visible(True)
    assert viewer.figure['layout']['margin'] == _LABELED_MARGINS

    viewer.set_x_axis_labels_visible(False)
    viewer.set_y_axis_labels_visible(False)
    assert viewer.figure['layout']['margin'] == _COMPACT_MARGINS


def test_layout_margins_profile_overrides_default_raster_margins() -> None:
    """Stack profiles should replace raster default margin tables."""
    profile = PlotlyLayoutMarginsProfile(
        with_axis_labels={'l': 60, 'r': 24, 't': 10, 'b': 40},
        compact={'l': 8, 'r': 8, 't': 8, 'b': 8},
        stabilize_axis_automargin=True,
    )
    viewer = _viewer_with_data(
        display_options=PlotlyRasterViewerDisplayOptions(layout_margins_profile=profile),
    )

    viewer.set_x_axis_labels_visible(True)
    viewer.set_y_axis_labels_visible(True)

    assert viewer.figure['layout']['margin'] == {'l': 60, 'r': 24, 't': 10, 'b': 40}
    assert viewer.figure['layout']['xaxis']['automargin'] is False
    assert viewer.figure['layout']['yaxis']['automargin'] is False


def test_axis_toggle_controls_titles_ticks_lines_and_grid() -> None:
    """Axis toggles should hide/show decorations independently with grid off."""
    viewer = _viewer_with_data()
    xaxis = viewer.figure['layout']['xaxis']
    yaxis = viewer.figure['layout']['yaxis']

    for axis in (xaxis, yaxis):
        assert axis['showticklabels'] is False
        assert axis['ticks'] == ''
        assert axis['showline'] is False
        assert axis['zeroline'] is False
        assert axis['showgrid'] is False

    viewer.set_x_axis_labels_visible(True)

    assert xaxis['showticklabels'] is True
    assert xaxis['ticks'] == 'outside'
    assert xaxis['showline'] is True
    assert xaxis['showgrid'] is False
    assert yaxis['showticklabels'] is False

    viewer.set_y_axis_labels_visible(True)

    assert yaxis['showticklabels'] is True
    assert yaxis['ticks'] == 'outside'
    assert yaxis['showline'] is True
    assert yaxis['showgrid'] is False


def test_init_axis_label_visibility_kwargs() -> None:
    """Constructor kwargs should set independent x/y axis label visibility."""
    viewer = _viewer_with_data(
        display_options=PlotlyRasterViewerDisplayOptions(
            show_x_axis_labels=True,
            show_y_axis_labels=False,
        ),
    )

    assert viewer.display_options.show_x_axis_labels is True
    assert viewer.display_options.show_y_axis_labels is False
    assert viewer.figure['layout']['xaxis']['showticklabels'] is True
    assert viewer.figure['layout']['yaxis']['showticklabels'] is False
    assert viewer.figure['layout']['margin'] == _LABELED_MARGINS


def test_axis_label_font_size_is_explicit() -> None:
    """Axis title and tick labels should use the shared default font size."""
    viewer = _viewer_with_data()
    viewer.set_x_axis_labels_visible(True)
    viewer.set_y_axis_labels_visible(True)

    for axis_name in ('xaxis', 'yaxis'):
        axis = viewer.figure['layout'][axis_name]
        assert axis['title']['font']['size'] == 11
        assert axis['tickfont']['size'] == 11


def test_plotly_theme_can_be_toggled_without_rebuilding_viewer() -> None:
    """Theme toggles should update Plotly layout colors in-place."""
    viewer = _viewer_with_data()

    viewer.set_dark_mode(True)
    layout = viewer.figure['layout']
    assert layout['paper_bgcolor'] == '#111827'
    assert layout['plot_bgcolor'] == '#111827'
    assert layout['font']['color'] == '#f9fafb'
    assert layout['xaxis']['gridcolor'] == '#374151'

    viewer.set_dark_mode(False)
    layout = viewer.figure['layout']
    assert layout['paper_bgcolor'] == 'white'
    assert layout['plot_bgcolor'] == 'white'
    assert layout['font']['color'] == '#111827'
    assert layout['xaxis']['gridcolor'] == '#e5e7eb'


def test_roi_visibility_toggle_sets_plotly_shape_visible_only() -> None:
    """ROI toggles should use Plotly shape visibility instead of deleting ROIs."""
    viewer = _viewer_with_data()
    viewer.set_rois([RectRoiOverlay(roi_id=1, x0=0, x1=1, y0=2, y1=3)])

    roi_shape = viewer.figure['layout']['shapes'][0]
    assert roi_shape['name'] == 'roi:1'
    assert roi_shape['visible'] is True

    viewer.set_roi_overlays_visible(False)

    roi_shape = viewer.figure['layout']['shapes'][0]
    assert roi_shape['name'] == 'roi:1'
    assert roi_shape['visible'] is False


def test_trace_visibility_toggle_preserves_trace_and_respects_overlay_visible() -> None:
    """Trace toggles should use Plotly trace visibility without deleting traces."""
    viewer = _viewer_with_data()
    viewer.set_trace_overlays([
        PlotlyTraceOverlay(trace_id='visible-trace', x=[1], y=[2], visible=True),
        PlotlyTraceOverlay(trace_id='hidden-trace', x=[3], y=[4], visible=False),
    ])

    traces = {trace['meta']['trace_id']: trace for trace in viewer.figure['data'][1:]}
    assert traces['visible-trace']['visible'] is True
    assert traces['hidden-trace']['visible'] is False

    viewer.set_trace_overlays_visible(False)
    traces = {trace['meta']['trace_id']: trace for trace in viewer.figure['data'][1:]}
    assert traces['visible-trace']['visible'] is False
    assert traces['hidden-trace']['visible'] is False

    viewer.set_trace_overlays_visible(True)
    traces = {trace['meta']['trace_id']: trace for trace in viewer.figure['data'][1:]}
    assert traces['visible-trace']['visible'] is True
    assert traces['hidden-trace']['visible'] is False


def test_context_menu_toggle_label_uses_check_prefix() -> None:
    """Context menu labels should show a check mark only when enabled."""
    assert PlotlyRasterViewerContextMenu._toggle_label('ROIs', True) == '✓ ROIs'
    assert PlotlyRasterViewerContextMenu._toggle_label('ROIs', False) == 'ROIs'


def test_show_roi_labels_defaults_to_true() -> None:
    """ROI labels should be visible by default."""
    assert PlotlyRasterViewerDisplayOptions().show_roi_labels is True


def test_roi_label_visibility_toggle_blanks_and_restores_label_text() -> None:
    """Hiding ROI labels should blank the shape label text without losing ROI state."""
    viewer = _viewer_with_data()
    viewer.set_rois([RectRoiOverlay(roi_id=1, x0=0, x1=1, y0=2, y1=3, label='1')])

    roi_shape = viewer.figure['layout']['shapes'][0]
    assert roi_shape['label'] == {'text': '1', 'textposition': 'top left'}

    viewer.set_roi_labels_visible(False)
    roi_shape = viewer.figure['layout']['shapes'][0]
    assert roi_shape['name'] == 'roi:1'
    assert roi_shape['label']['text'] == ''

    viewer.set_roi_labels_visible(True)
    roi_shape = viewer.figure['layout']['shapes'][0]
    assert roi_shape['label']['text'] == '1'


def test_hover_info_defaults_to_skip_on_initial_data_set() -> None:
    """Default ``show_hover_info=False`` writes ``hoverinfo='skip'`` on the raster trace."""
    viewer = _viewer_with_data()
    trace0 = viewer.figure['data'][0]
    assert trace0['hoverinfo'] == 'skip'


def test_hover_info_setter_updates_local_figure_dict_in_both_directions() -> None:
    """Toggling hover info should flip the raster trace's ``hoverinfo`` value."""
    viewer = _viewer_with_data()
    assert viewer.figure['data'][0]['hoverinfo'] == 'skip'

    viewer.set_hover_info_visible(True)
    assert viewer.display_options.show_hover_info is True
    assert viewer.figure['data'][0]['hoverinfo'] == 'all'

    viewer.set_hover_info_visible(False)
    assert viewer.display_options.show_hover_info is False
    assert viewer.figure['data'][0]['hoverinfo'] == 'skip'
