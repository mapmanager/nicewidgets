"""Tests for ``ContrastWidget`` user handlers and ext setters."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from nicewidgets.contrast_widget.colorscales import (
    colorscale_option_value_to_label,
)
from nicewidgets.contrast_widget.contrast_widget import (
    DEFAULT_LUT,
    DEFAULT_RANGE_MAX,
    DEFAULT_RANGE_MIN,
    ContrastWidget,
)
from nicewidgets.contrast_widget.intent import ContrastChangedIntent


@pytest.fixture
def widget() -> ContrastWidget:
    return ContrastWidget()


def test_construct_defaults(widget: ContrastWidget) -> None:
    """Widget should construct with documented defaults and no emit on init."""
    assert widget.get_color_lut() == DEFAULT_LUT
    assert widget.get_range() == (DEFAULT_RANGE_MIN, DEFAULT_RANGE_MAX)
    assert widget.get_image_bounds() == (DEFAULT_RANGE_MIN, DEFAULT_RANGE_MAX)
    assert widget.get_image() is None


def test_lut_select_uses_value_to_label_mapping(widget: ContrastWidget) -> None:
    """ui.select.options is a {value: label} dict so dropdown shows labels.

    The wire value remains the internal identifier (e.g. ``'inverted_grays'``)
    while the displayed label is human-friendly (``'Inverted Gray'``).
    """
    options = widget._lut_select.options
    assert options == colorscale_option_value_to_label()
    assert options['inverted_grays'] == 'Inverted Gray'
    assert options['Gray'] == 'Gray'
    assert options['Plasma'] == 'Plasma'


def test_set_lut_ext_does_not_emit() -> None:
    """`set_lut_ext` updates state and never invokes the intent callback."""
    seen: list[ContrastChangedIntent] = []
    w = ContrastWidget(on_intent=seen.append)
    w.set_lut_ext('Viridis')
    assert w.get_color_lut() == 'Viridis'
    assert seen == []


def test_set_range_ext_does_not_emit_and_swaps_inverted_pair() -> None:
    """`set_range_ext` swaps inverted pairs and never emits."""
    seen: list[ContrastChangedIntent] = []
    w = ContrastWidget(on_intent=seen.append)
    w.set_range_ext(value_min=80, value_max=20)
    assert w.get_range() == (20, 80)
    assert seen == []


def test_set_image_ext_updates_bounds_without_emit() -> None:
    """`set_image_ext` recomputes the range bounds without emitting."""
    seen: list[ContrastChangedIntent] = []
    w = ContrastWidget(on_intent=seen.append)
    img = np.array([[10, 200], [50, 150]], dtype=np.uint16)
    w.set_image_ext(img)
    assert w.get_image_bounds() == (10, 200)
    assert w.get_image() is img
    assert seen == []


def test_set_image_ext_none_resets_bounds() -> None:
    """Passing ``None`` reverts bounds to defaults and clears the cached image."""
    w = ContrastWidget()
    w.set_image_ext(np.array([[1, 9]], dtype=np.uint16))
    w.set_image_ext(None)
    assert w.get_image_bounds() == (DEFAULT_RANGE_MIN, DEFAULT_RANGE_MAX)
    assert w.get_image() is None


def test_set_enabled_ext_disables_user_intents() -> None:
    """Disabling the widget suppresses user-driven emits."""
    seen: list[ContrastChangedIntent] = []
    w = ContrastWidget(on_intent=seen.append)
    w.set_enabled_ext(False)
    w._on_range_change(SimpleNamespace(value={'min': 5, 'max': 90}))
    assert seen == []
    w.set_enabled_ext(True)
    w._on_range_change(SimpleNamespace(value={'min': 5, 'max': 90}))
    assert len(seen) == 1
    assert seen[0] == ContrastChangedIntent(
        color_lut=DEFAULT_LUT, value_min=5, value_max=90
    )


def test_lut_change_emits_full_state() -> None:
    """User LUT changes emit one intent carrying the full current state.

    Assigning to ``_lut_select.value`` fires the registered ``on_change``
    handler in NiceGUI; an explicit handler call would double-emit.
    """
    seen: list[ContrastChangedIntent] = []
    w = ContrastWidget(on_intent=seen.append)
    w.set_range_ext(value_min=10, value_max=240)
    w._lut_select.value = 'Plasma'
    assert seen == [
        ContrastChangedIntent(color_lut='Plasma', value_min=10, value_max=240)
    ]


def test_range_change_emits_full_state_and_swaps() -> None:
    """User range changes emit a single intent with swapped values when inverted."""
    seen: list[ContrastChangedIntent] = []
    w = ContrastWidget(on_intent=seen.append)
    w.set_lut_ext('Hot')
    w._on_range_change(SimpleNamespace(value={'min': 200, 'max': 50}))
    assert seen == [
        ContrastChangedIntent(color_lut='Hot', value_min=50, value_max=200)
    ]


def test_auto_button_uses_callback_and_emits_once() -> None:
    """Auto computes via callback, updates the range, and emits exactly once."""
    seen: list[ContrastChangedIntent] = []
    img = np.array([[0, 100, 200, 255]], dtype=np.uint8)

    def auto(_img: np.ndarray) -> tuple[int, int]:
        return 5, 250

    w = ContrastWidget(on_intent=seen.append, auto_contrast_callback=auto)
    w.set_image_ext(img)
    w.set_lut_ext('Green')
    w._on_auto_click()
    assert w.get_range() == (5, 250)
    assert seen == [
        ContrastChangedIntent(color_lut='Green', value_min=5, value_max=250, from_auto=True)
    ]


def test_auto_button_noop_without_image_or_callback() -> None:
    """Auto does nothing (and emits nothing) when image or callback is missing."""
    seen: list[ContrastChangedIntent] = []
    w_no_cb = ContrastWidget(on_intent=seen.append)
    w_no_cb.set_image_ext(np.array([[0, 255]], dtype=np.uint8))
    w_no_cb._on_auto_click()
    assert seen == []

    w_no_img = ContrastWidget(
        on_intent=seen.append, auto_contrast_callback=lambda _i: (10, 20)
    )
    w_no_img._on_auto_click()
    assert seen == []


def test_auto_swaps_inverted_callback_result() -> None:
    """Auto callback returning min > max is swapped before emit."""
    seen: list[ContrastChangedIntent] = []
    w = ContrastWidget(
        on_intent=seen.append, auto_contrast_callback=lambda _i: (240, 8)
    )
    w.set_image_ext(np.array([[1, 200]], dtype=np.uint16))
    w._on_auto_click()
    assert w.get_range() == (8, 240)
    assert seen == [
        ContrastChangedIntent(color_lut=DEFAULT_LUT, value_min=8, value_max=240, from_auto=True)
    ]


def test_callback_exception_suppressed_and_no_emit() -> None:
    """Auto callback exceptions are caught; widget remains stable and emits nothing."""
    seen: list[ContrastChangedIntent] = []

    def bad(_img: np.ndarray) -> tuple[int, int]:
        raise RuntimeError('boom')

    w = ContrastWidget(on_intent=seen.append, auto_contrast_callback=bad)
    w.set_image_ext(np.array([[1, 9]], dtype=np.uint16))
    w._on_auto_click()
    assert seen == []


def test_range_slider_uses_bounded_flex_classes(widget: ContrastWidget) -> None:
    """Range slider must flex but stay bounded so it does not dominate the row.

    The previous fix removed a fixed ``w-56`` and made the slider unbounded,
    which then pushed ``_max_label`` onto the next line whenever the host row
    narrowed. The widget now ships ``flex-1 min-w-32 max-w-64`` on the range
    so it grows with available space but never swallows its companion labels.
    """
    classes = widget._range.classes
    assert 'flex-1' in classes
    assert 'min-w-32' in classes
    assert 'max-w-48' in classes


def test_range_companion_labels_keep_fixed_width(widget: ContrastWidget) -> None:
    """Min/max labels keep their narrow fixed width so the group stays compact."""
    assert 'w-8' in widget._min_label.classes
    assert 'w-8' in widget._max_label.classes
