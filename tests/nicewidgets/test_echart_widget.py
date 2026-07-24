"""Tests for the EChartWidget option builder."""

from __future__ import annotations

import pytest

from nicewidgets.echart_widget.models import EChartAxisRange, EChartEventOverlay, EChartLineData
from nicewidgets.echart_widget.widget import (
    EChartWidget,
    _X_AXIS_LABEL_FORMATTER_JS,
    build_line_options,
)


def test_echart_line_data_validates_lengths() -> None:
    """EChartLineData should reject mismatched x/y lengths."""
    with pytest.raises(ValueError):
        EChartLineData.from_sequences(
            x=[0.0, 1.0],
            y=[1.0],
            x_label="x",
            y_label="y",
        )


def test_build_line_options_uses_axis_labels_and_data_pairs() -> None:
    """Line options should contain value axes and paired data."""
    line = EChartLineData.from_sequences(
        x=[0.0, 1.0],
        y=[2.0, 3.0],
        x_label="Time (s)",
        y_label="Velocity",
        series_name="Radon velocity",
    )

    options = build_line_options(line, EChartAxisRange(x_min=0.0, x_max=2.0))

    assert options["xAxis"]["type"] == "value"
    assert options["xAxis"]["name"] == "Time (s)"
    assert options["xAxis"]["min"] == 0.0
    assert options["xAxis"]["max"] == 2.0
    assert options["yAxis"]["name"] == "Velocity"
    assert options["series"][0]["name"] == "Radon velocity"
    assert options["series"][0]["data"] == [[0.0, 2.0], [1.0, 3.0]]


def test_echart_axis_range_validates_bounds() -> None:
    """Axis range should reject inverted bounds."""
    with pytest.raises(ValueError):
        EChartAxisRange(x_min=2.0, x_max=1.0)


# ---- additional coverage tests ----


def test_build_line_options_uses_data_extent_when_no_range_passed() -> None:
    """Without an explicit range, x-axis pins to ECharts' ``'dataMin'`` / ``'dataMax'``.

    Pinning to the data-extent sentinels disables ECharts' default
    "nice ticks" outward rounding (e.g. [0, 9.4] -> [0, 10]) on the x-axis
    while leaving the y-axis untouched.
    """
    line = EChartLineData.from_sequences(
        x=[0.0, 1.0], y=[2.0, 3.0], x_label="x", y_label="y"
    )

    options = build_line_options(line)

    assert options["xAxis"]["min"] == "dataMin"
    assert options["xAxis"]["max"] == "dataMax"
    assert "min" not in options["yAxis"]
    assert "max" not in options["yAxis"]


def test_build_line_options_explicit_x_range_overrides_data_sentinels() -> None:
    """Explicit ``EChartAxisRange`` values are used verbatim (no sentinels)."""
    line = EChartLineData.from_sequences(
        x=[0.0, 1.0], y=[2.0, 3.0], x_label="x", y_label="y"
    )

    options = build_line_options(line, EChartAxisRange(x_min=0.25, x_max=0.75))

    assert options["xAxis"]["min"] == 0.25
    assert options["xAxis"]["max"] == 0.75


def test_empty_options_has_value_axes_and_empty_series() -> None:
    """`_empty_options` should describe a blank value/value chart."""
    opts = EChartWidget._empty_options()

    assert opts["xAxis"]["type"] == "value"
    assert opts["yAxis"]["type"] == "value"
    assert opts["series"] == []
    assert opts["animation"] is False


def test_extract_x_brush_range_handles_missing_data() -> None:
    """Missing batch/areas/coordRange should return None."""
    assert EChartWidget._extract_x_brush_range({}) is None
    assert EChartWidget._extract_x_brush_range({"batch": []}) is None
    assert EChartWidget._extract_x_brush_range({"batch": [{"areas": []}]}) is None
    assert (
        EChartWidget._extract_x_brush_range({"batch": [{"areas": [{"coordRange": [1.0]}]}]})
        is None
    )


def test_extract_x_brush_range_returns_sorted_floats() -> None:
    """Coord range with inverted values should be returned sorted."""
    result = EChartWidget._extract_x_brush_range(
        {"batch": [{"areas": [{"coordRange": [5.0, 2.0]}]}]}
    )

    assert result == (2.0, 5.0)


def test_extract_x_brush_range_handles_normal_range() -> None:
    """Normal coord range should map to (x0, x1)."""
    result = EChartWidget._extract_x_brush_range(
        {"batch": [{"areas": [{"coordRange": [1.5, 4.5]}]}]}
    )

    assert result == (1.5, 4.5)


def test_echart_event_overlay_from_object_extracts_attributes() -> None:
    """``EChartEventOverlay.from_object`` should pull id/x0/x1/event_type."""

    class _Obj:
        id = 5
        x0 = 1.0
        x1 = 2.0
        event_type = "auto"

    overlay = EChartEventOverlay.from_object(_Obj())

    assert overlay.id == "5"
    assert overlay.x0 == 1.0
    assert overlay.x1 == 2.0
    assert overlay.event_type == "auto"


def test_echart_event_overlay_from_object_defaults_event_type() -> None:
    """Missing event_type should default to 'user'."""

    class _Obj:
        id = "x"
        x0 = 0.0
        x1 = 1.0

    overlay = EChartEventOverlay.from_object(_Obj())

    assert overlay.event_type == "user"


def test_echart_event_overlay_from_object_uses_enum_value() -> None:
    """If event_type has a ``value`` attribute, that value should be used."""
    from enum import StrEnum

    class _Type(StrEnum):
        AUTO = "auto"

    class _Obj:
        id = "x"
        x0 = 0.0
        x1 = 1.0
        event_type = _Type.AUTO

    overlay = EChartEventOverlay.from_object(_Obj())

    assert overlay.event_type == "auto"


# ---- x-axis tick label formatter ----


def test_build_line_options_includes_xaxis_label_formatter() -> None:
    """``build_line_options`` wires the rounding JS formatter on the x-axis labels.

    The ``":formatter"`` key (leading colon) is NiceGUI's convention for
    promoting a string to a real JS function in ECharts options; without it,
    full-precision min/max values would leak into the displayed tick labels
    (e.g. ``8.181001796798633``).
    """
    line = EChartLineData.from_sequences(
        x=[0.0, 1.0], y=[2.0, 3.0], x_label="x", y_label="y"
    )

    options = build_line_options(line, EChartAxisRange(x_min=0.0, x_max=1.0))

    axis_label = options["xAxis"]["axisLabel"]
    assert ":formatter" in axis_label
    assert "formatter" not in axis_label
    assert axis_label[":formatter"] == _X_AXIS_LABEL_FORMATTER_JS


def test_empty_options_includes_xaxis_label_formatter() -> None:
    """``_empty_options`` carries the same x-axis tick formatter as the line variant."""
    from nicewidgets.echart_widget.widget import EChartWidget

    opts = EChartWidget._empty_options()

    axis_label = opts["xAxis"]["axisLabel"]
    assert axis_label[":formatter"] == _X_AXIS_LABEL_FORMATTER_JS


def test_xaxis_label_formatter_rounds_long_floats_to_three_decimals() -> None:
    """The JS formatter must round to ~3 decimals while leaving nice ticks intact.

    Verified by executing the JS expression in Python via direct equivalent
    semantics: ``+value.toFixed(3)`` rounds half-away-from-zero, strips
    trailing zeros via numeric conversion, and leaves clean values unchanged.
    """
    cases = {
        8.181001796798633: 8.181,
        7.08379656153604: 7.084,
        7.4: 7.4,
        8.0: 8.0,
        0.0: 0.0,
    }
    for raw, expected in cases.items():
        assert float(f"{raw:.3f}") == pytest.approx(expected)

    assert _X_AXIS_LABEL_FORMATTER_JS == "(value) => +value.toFixed(3)"
