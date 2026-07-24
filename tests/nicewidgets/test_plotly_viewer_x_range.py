"""Tests for :class:`PlotlyRasterViewer` x-range producer hook + echo dedup."""

from __future__ import annotations

import asyncio
import sys
import types

import numpy as np

if 'nicegui' not in sys.modules:
    fake_nicegui = types.ModuleType('nicegui')
    fake_nicegui.ui = types.SimpleNamespace()
    sys.modules['nicegui'] = fake_nicegui

from nicewidgets.raster_viewer.backend.image_model import RasterGridSpec
from nicewidgets.raster_viewer.frontend.plotly_viewer import PlotlyRasterViewer


def _viewer_with_data(callback):
    """Build a viewer with a small raster and the supplied x-range callback.

    ``set_data`` pre-populates ``_last_applied_x_range`` with the data extent
    so the follow-up Plotly relayout (post-uirevision-rotation) is suppressed
    by value-based dedup rather than a one-shot guard. Tests that want to
    exercise a different baseline override ``_last_applied_x_range`` directly.
    """
    viewer = PlotlyRasterViewer(on_x_range_changed=callback)
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='s', y_unit='um')
    asyncio.run(viewer.set_data(data, grid=grid))
    return viewer


def test_relayout_with_xrange_invokes_callback_with_floats() -> None:
    """A merged relayout payload should produce a single callback emission."""
    seen: list[tuple[float | None, float | None]] = []
    viewer = _viewer_with_data(lambda lo, hi: seen.append((lo, hi)))

    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 1.0,
        'xaxis.range[1]': 4.5,
    })
    assert seen == [(1.0, 4.5)]


def test_partial_relayout_without_xrange_does_not_invoke() -> None:
    """A relayout payload without ``xaxis.range`` entries is ignored."""
    seen: list[tuple[float | None, float | None]] = []
    viewer = _viewer_with_data(lambda lo, hi: seen.append((lo, hi)))

    viewer._emit_x_range_from_relayout({'yaxis.range[0]': 0.0, 'yaxis.range[1]': 9.0})
    assert seen == []


def test_relayout_echoing_last_apply_is_suppressed() -> None:
    """Echo of a programmatic apply should not re-fire the callback."""
    seen: list[tuple[float | None, float | None]] = []
    viewer = _viewer_with_data(lambda lo, hi: seen.append((lo, hi)))

    viewer._last_applied_x_range = (1.0, 4.5)
    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 1.0,
        'xaxis.range[1]': 4.5,
    })
    assert seen == []

    # A new user pan does fire.
    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 2.0,
        'xaxis.range[1]': 5.0,
    })
    assert seen == [(2.0, 5.0)]


def test_no_callback_means_no_emission_and_no_error() -> None:
    """Constructing without a callback must not error on relayout."""
    viewer = _viewer_with_data(callback=None)
    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 0.5,
        'xaxis.range[1]': 1.5,
    })


def test_reset_x_axis_range_clears_echo_suppressor() -> None:
    """``reset_x_axis_range`` sets the echo state so any future range will fire."""
    seen: list[tuple[float | None, float | None]] = []
    viewer = _viewer_with_data(lambda lo, hi: seen.append((lo, hi)))

    viewer._last_applied_x_range = (1.0, 2.0)
    viewer.reset_x_axis_range()
    assert viewer._last_applied_x_range == (None, None)
    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 1.0,
        'xaxis.range[1]': 2.0,
    })
    assert seen == [(1.0, 2.0)]


def test_set_data_pins_last_applied_to_data_extent() -> None:
    """``set_data`` pre-populates ``_last_applied_x_range`` with the data extent.

    Plotly fires a fresh ``plotly_relayout`` after ``_uirevision`` rotates;
    that relayout carries the new auto-ranged data extent. Pinning
    ``_last_applied_x_range`` to the same value lets ``_is_x_range_echo``
    suppress it by value, without an extra one-shot guard that could swallow
    the user's first real gesture.
    """
    viewer = PlotlyRasterViewer(on_x_range_changed=lambda lo, hi: None)
    data = np.arange(100, dtype=np.float32).reshape(10, 10)
    grid = RasterGridSpec(dx=1.0, dy=1.0, x_unit='s', y_unit='um')
    asyncio.run(viewer.set_data(data, grid=grid))
    assert viewer._last_applied_x_range == (0.0, 10.0)


def test_post_set_data_data_extent_relayout_is_suppressed_by_value() -> None:
    """The follow-up relayout carrying the data extent is suppressed by value.

    After ``set_data``, Plotly may emit a relayout with the new auto-ranged
    data extent; that payload matches ``_last_applied_x_range`` and must
    not re-fire ``on_x_range_changed``.
    """
    seen: list[tuple[float | None, float | None]] = []
    viewer = _viewer_with_data(lambda lo, hi: seen.append((lo, hi)))

    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 0.0,
        'xaxis.range[1]': 10.0,
    })
    assert seen == []


def test_first_user_relayout_after_set_data_fires() -> None:
    """The user's first relayout after ``set_data`` must reach the callback.

    Regression: the previous one-shot guard could swallow the user's first
    pan/zoom when Plotly emitted only ``autosize`` (no ``xaxis.range``) for
    the post-uirevision-rotation relayout. Value-based dedup never blocks a
    relayout whose ranges differ from the recorded data extent.
    """
    seen: list[tuple[float | None, float | None]] = []
    viewer = _viewer_with_data(lambda lo, hi: seen.append((lo, hi)))

    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 2.0,
        'xaxis.range[1]': 6.0,
    })
    assert seen == [(2.0, 6.0)]


def test_non_xrange_relayout_does_not_affect_dedup_baseline() -> None:
    """A relayout without ``xaxis.range`` keys leaves dedup state untouched.

    The early-return inside ``_on_plotly_relayout`` filters such payloads
    before reaching ``_emit_x_range_from_relayout``; dedup baseline must not
    be consumed or rotated by them. This guarantees the first real user
    gesture still emits.
    """
    seen: list[tuple[float | None, float | None]] = []
    viewer = _viewer_with_data(lambda lo, hi: seen.append((lo, hi)))
    baseline = viewer._last_applied_x_range

    viewer._emit_x_range_from_relayout({'yaxis.range[0]': 0.0, 'yaxis.range[1]': 9.0})
    assert seen == []
    assert viewer._last_applied_x_range == baseline

    viewer._emit_x_range_from_relayout({
        'xaxis.range[0]': 3.0,
        'xaxis.range[1]': 5.0,
    })
    assert seen == [(3.0, 5.0)]
