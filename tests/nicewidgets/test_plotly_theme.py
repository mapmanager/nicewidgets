"""Tests for shared Plotly layout themes and pool figure generation."""

from __future__ import annotations

import pandas as pd

from nicewidgets.nicepool.dataframe_processor import DataFrameProcessor
from nicewidgets.nicepool.figure_generator import FigureGenerator
from nicewidgets.nicepool.plot_pool_controller import PlotPoolConfig, PlotPoolController
from nicewidgets.nicepool.plot_state import PlotState, PlotType
from nicewidgets.nicepool.pre_filter_conventions import PRE_FILTER_NONE
from nicewidgets.plotly_theme import (
    apply_plotly_theme_to_layout,
    normalize_plotly_theme,
    theme_for_name,
)


def test_normalize_plotly_theme() -> None:
    """Theme normalization should accept only light and dark."""
    assert normalize_plotly_theme('dark') == 'dark'
    assert normalize_plotly_theme('DARK') == 'dark'
    assert normalize_plotly_theme('light') == 'light'
    assert normalize_plotly_theme('unknown') == 'light'


def test_apply_plotly_theme_to_layout_sets_axis_colors() -> None:
    """Layout theming should update background, font, and axis colors."""
    layout: dict[str, object] = {}

    apply_plotly_theme_to_layout(layout, 'dark')

    dark = theme_for_name('dark')
    assert layout['paper_bgcolor'] == dark.paper_bgcolor
    assert layout['plot_bgcolor'] == dark.plot_bgcolor
    assert layout['font'] == {'color': dark.font_color}
    xaxis = layout['xaxis']
    assert isinstance(xaxis, dict)
    assert xaxis['gridcolor'] == dark.grid_color


def test_figure_generator_applies_dark_layout_theme() -> None:
    """FigureGenerator should apply layout theme at the end of make_figure."""
    df = pd.DataFrame(
        [
            {"pool_row_id": "a", "channel": 0, "x": 1.0, "y": 2.0},
            {"pool_row_id": "b", "channel": 1, "x": 2.0, "y": 4.0},
        ]
    )
    processor = DataFrameProcessor(df, pre_filter_columns=["channel"], unique_row_id_col="pool_row_id")
    generator = FigureGenerator(processor, unique_row_id_col="pool_row_id")
    generator.set_dark_mode(True)
    state = PlotState(
        pre_filter={"channel": PRE_FILTER_NONE},
        xcol="x",
        ycol="y",
        plot_type=PlotType.SCATTER,
    )

    figure, _summary = generator.make_figure(processor.filter_by_pre_filters(state.pre_filter), state)

    assert figure["layout"]["paper_bgcolor"] == "#111827"
    assert figure["layout"]["font"]["color"] == "#f9fafb"


def test_plot_pool_controller_set_dark_mode_updates_figure_generator() -> None:
    """PlotPoolController should delegate theme changes to FigureGenerator."""
    df = pd.DataFrame([{"path": "a", "x": 1.0, "y": 2.0}])
    controller = PlotPoolController(
        df,
        config=PlotPoolConfig(enable_config_persistence=False, pre_filter_columns=[]),
    )

    controller.set_dark_mode(True)

    figure, _summary = controller.figure_generator.make_figure(df, controller.plot_states[0])
    assert figure["layout"]["paper_bgcolor"] == "#111827"
