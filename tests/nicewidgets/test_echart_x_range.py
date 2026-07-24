"""Tests for :class:`EChartWidget` x-range producer hook + echo dedup.

The widget's ``__init__`` requires ``ui.echart`` (NiceGUI), so these tests
construct the widget via ``__new__`` and exercise the pure-Python helpers
directly. The behavior we care about (datazoom decoding, echo dedup) is
isolated from the NiceGUI layer.
"""

from __future__ import annotations

from nicewidgets.echart_widget.models import EChartAxisRange, EChartLineData
from nicewidgets.echart_widget.widget import EChartWidget


def _make_widget(
    *,
    on_x_range_changed=None,
    line_data: EChartLineData | None = None,
) -> EChartWidget:
    """Build an EChartWidget without invoking ``__init__`` (no NiceGUI required)."""
    w = EChartWidget.__new__(EChartWidget)
    w._line_data = line_data
    w._x_range = EChartAxisRange()
    w._on_x_range_changed = on_x_range_changed
    w._last_applied_x_range = None
    return w


def test_extract_datazoom_uses_start_value_end_value_when_present() -> None:
    """Absolute axis values (``startValue``/``endValue``) take priority."""
    w = _make_widget()
    result = w._extract_x_datazoom_range({'startValue': 1.5, 'endValue': 4.0})
    assert result == (1.5, 4.0)


def test_extract_datazoom_unwraps_batch_wrapper() -> None:
    """ECharts datazoom emits ``batch=[{...}]``; the inner payload is read."""
    w = _make_widget()
    result = w._extract_x_datazoom_range(
        {'batch': [{'startValue': 2.0, 'endValue': 5.0}]}
    )
    assert result == (2.0, 5.0)


def test_extract_datazoom_percent_uses_line_data() -> None:
    """Percent ``start``/``end`` resolve against line-data x extent."""
    line = EChartLineData.from_sequences(
        x=[10.0, 20.0], y=[1.0, 2.0], x_label='x', y_label='y'
    )
    w = _make_widget(line_data=line)
    result = w._extract_x_datazoom_range({'start': 0.0, 'end': 50.0})
    assert result == (10.0, 15.0)


def test_extract_datazoom_percent_full_range_returns_auto_none() -> None:
    """``start=0, end=100`` collapses to ``(None, None)`` (auto)."""
    line = EChartLineData.from_sequences(
        x=[0.0, 10.0], y=[1.0, 2.0], x_label='x', y_label='y'
    )
    w = _make_widget(line_data=line)
    assert w._extract_x_datazoom_range({'start': 0.0, 'end': 100.0}) == (None, None)


def test_extract_datazoom_percent_without_line_data_returns_none() -> None:
    """Percent payload with no line data cannot be resolved."""
    w = _make_widget(line_data=None)
    assert w._extract_x_datazoom_range({'start': 0.0, 'end': 50.0}) is None


def test_datazoom_invokes_callback_once_for_user_change() -> None:
    """A non-echo datazoom triggers the callback exactly once."""
    seen: list[tuple[float | None, float | None]] = []
    w = _make_widget(on_x_range_changed=lambda lo, hi: seen.append((lo, hi)))

    class _Event:
        args = {'startValue': 1.0, 'endValue': 4.0}

    w._on_datazoom(_Event())
    assert seen == [(1.0, 4.0)]


def test_datazoom_echo_after_set_x_axis_limits_is_suppressed() -> None:
    """Echo (matching last applied) does not re-fire the callback."""
    seen: list[tuple[float | None, float | None]] = []
    w = _make_widget(on_x_range_changed=lambda lo, hi: seen.append((lo, hi)))
    w._last_applied_x_range = (1.0, 4.0)

    class _Event:
        args = {'startValue': 1.0, 'endValue': 4.0}

    w._on_datazoom(_Event())
    assert seen == []

    class _Event2:
        args = {'startValue': 2.0, 'endValue': 5.0}

    w._on_datazoom(_Event2())
    assert seen == [(2.0, 5.0)]


def test_datazoom_suppressed_when_logical_range_unchanged() -> None:
    """Duplicate datazoom for the current logical range must not re-fire."""
    from nicewidgets.echart_widget.models import EChartAxisRange

    seen: list[tuple[float | None, float | None]] = []
    w = _make_widget(on_x_range_changed=lambda lo, hi: seen.append((lo, hi)))
    w._x_range = EChartAxisRange(x_min=1.0, x_max=4.0)

    class _Event:
        args = {"startValue": 1.0, "endValue": 4.0}

    w._on_datazoom(_Event())
    assert seen == []


def test_set_x_axis_limits_skips_apply_when_range_unchanged() -> None:
    """Programmatic limits should no-op when the logical range is already set."""
    from nicewidgets.echart_widget.models import EChartAxisRange

    w = EChartWidget.__new__(EChartWidget)
    w._line_data = None
    w._x_range = EChartAxisRange(x_min=2.0, x_max=6.0)
    w._last_applied_x_range = None
    w.apply = lambda: (_ for _ in ()).throw(AssertionError("apply should not run"))  # type: ignore[method-assign]

    w.set_x_axis_limits(2.0, 6.0)


def test_datazoom_with_no_callback_is_a_noop() -> None:
    """Without a callback, datazoom handling must not raise."""
    w = _make_widget(on_x_range_changed=None)

    class _Event:
        args = {'startValue': 0.5, 'endValue': 1.5}

    w._on_datazoom(_Event())
