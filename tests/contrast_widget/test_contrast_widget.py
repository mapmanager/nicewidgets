"""Tests for ContrastWidget and histogram module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from nicewidgets.contrast_widget.contrast_widget import ContrastParams, ContrastWidget
from nicewidgets.contrast_widget.histogram import histogram_plot_plotly


# --- ContrastParams ---


def test_contrast_params_frozen() -> None:
    """ContrastParams is immutable."""
    p = ContrastParams(colorscale="Gray", zmin=0, zmax=255)
    with pytest.raises(Exception):  # FrozenInstanceError
        p.zmin = 10  # type: ignore[misc]


# --- histogram_plot_plotly ---


def test_histogram_returns_dict_empty_image() -> None:
    """histogram_plot_plotly(None) returns a valid Plotly dict."""
    d = histogram_plot_plotly(None)
    assert isinstance(d, dict)
    assert "data" in d
    assert "layout" in d


def test_histogram_returns_dict_with_image(sample_image: np.ndarray) -> None:
    """histogram_plot_plotly with image returns dict with expected structure."""
    d = histogram_plot_plotly(
        sample_image,
        zmin=10,
        zmax=200,
        log_scale=True,
        theme="light",
    )
    assert isinstance(d, dict)
    assert "data" in d
    assert "layout" in d
    assert len(d["data"]) >= 1
    assert "xaxis" in d["layout"] or "xaxis" in str(d["layout"])


def test_histogram_with_log_and_linear(sample_image: np.ndarray) -> None:
    """histogram supports log_scale=True and False."""
    d_log = histogram_plot_plotly(sample_image, log_scale=True)
    d_linear = histogram_plot_plotly(sample_image, log_scale=False)
    assert "data" in d_log and "data" in d_linear
    # Layout should differ in yaxis type
    layout_log = d_log.get("layout", {})
    layout_linear = d_linear.get("layout", {})
    yaxis_log = layout_log.get("yaxis", {})
    yaxis_linear = layout_linear.get("yaxis", {})
    if isinstance(yaxis_log, dict) and isinstance(yaxis_linear, dict):
        assert yaxis_log.get("type") == "log"
        assert yaxis_linear.get("type") == "linear"


# --- ContrastWidget (mocked UI) ---


def test_contrast_widget_emits_on_slider_change() -> None:
    """ContrastWidget emits on_change with correct ContrastParams when sliders change."""
    received: list[ContrastParams] = []

    def on_change(p: ContrastParams) -> None:
        received.append(p)

    widget = ContrastWidget(on_change=on_change)
    widget._min_slider = MagicMock()
    widget._min_slider.value = 10
    widget._max_slider = MagicMock()
    widget._max_slider.value = 100
    widget._display_params = ContrastParams(colorscale="Gray", zmin=0, zmax=255)
    widget._updating_programmatically = False

    widget._on_slider_change()

    assert len(received) == 1
    assert received[0].zmin == 10
    assert received[0].zmax == 100
    assert received[0].colorscale == "Gray"


def test_contrast_widget_emits_on_colorscale_change() -> None:
    """ContrastWidget emits on_change when colorscale changes."""
    received: list[ContrastParams] = []

    def on_change(p: ContrastParams) -> None:
        received.append(p)

    widget = ContrastWidget(on_change=on_change)
    widget._colorscale_select = MagicMock()
    widget._colorscale_select.value = "Viridis"
    widget._display_params = ContrastParams(colorscale="Gray", zmin=0, zmax=255)
    widget._updating_programmatically = False

    widget._on_colorscale_change()

    assert len(received) == 1
    assert received[0].colorscale == "Viridis"


def test_contrast_widget_zmin_zmax_ordering() -> None:
    """When min > max, widget corrects and emits valid params."""
    received: list[ContrastParams] = []

    def on_change(p: ContrastParams) -> None:
        received.append(p)

    widget = ContrastWidget(on_change=on_change)
    widget._min_slider = MagicMock()
    widget._min_slider.value = 150  # min moved past max
    widget._max_slider = MagicMock()
    widget._max_slider.value = 100
    widget._min_value_label = MagicMock()
    widget._max_value_label = MagicMock()
    widget._display_params = ContrastParams(colorscale="Gray", zmin=0, zmax=255)
    widget._updating_programmatically = False

    widget._on_slider_change()

    assert len(received) == 1
    assert received[0].zmin <= received[0].zmax


def test_contrast_widget_set_params_updates_internal_state() -> None:
    """set_params updates _display_params (no UI in unit test)."""
    widget = ContrastWidget(on_change=lambda p: None)
    widget._colorscale_select = MagicMock()
    widget._min_slider = MagicMock()
    widget._max_slider = MagicMock()
    widget._min_value_label = MagicMock()
    widget._max_value_label = MagicMock()

    params = ContrastParams(colorscale="Viridis", zmin=20, zmax=180)
    widget._set_params_impl(params)

    assert widget._display_params == params
    assert widget._colorscale_select.value == "Viridis"
    assert widget._min_slider.value == 20
    assert widget._max_slider.value == 180


def test_contrast_widget_set_image_updates_slider_range(sample_image: np.ndarray) -> None:
    """set_image with array updates slider max and _display_params."""
    widget = ContrastWidget(on_change=lambda p: None)
    widget._min_slider = MagicMock()
    widget._max_slider = MagicMock()
    widget._min_value_label = MagicMock()
    widget._max_value_label = MagicMock()
    widget._display_params = ContrastParams(colorscale="Gray", zmin=0, zmax=255)

    widget._set_image_impl(sample_image)

    image_max = int(np.max(sample_image))
    assert widget._display_params is not None
    assert widget._display_params.zmax == image_max
    assert widget._max_slider.value == image_max
    widget._max_slider.props.assert_any_call(f"max={image_max}")
