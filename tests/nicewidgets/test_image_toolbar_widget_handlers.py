"""Headless tests for ImageToolbarWidget setter, parser, and click handlers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nicewidgets.image_toolbar_widget.image_toolbar_widget import (
    ImageToolbarMode,
    ImageToolbarWidget,
)
from nicewidgets.image_toolbar_widget.intent import (
    ImageToolbarRoiAddRequestIntent,
    ImageToolbarRoiApplyFullHeightIntent,
    ImageToolbarRoiApplyFullWidthIntent,
    ImageToolbarRoiDeleteRequestIntent,
    ImageToolbarRoiEditCancelIntent,
    ImageToolbarRoiEditStartIntent,
    ImageToolbarRoiEditSubmitIntent,
    ImageToolbarSelectChannelIntent,
)


@pytest.fixture
def hosted_toolbar() -> tuple[ImageToolbarWidget, list[object]]:
    """Return a toolbar wired to an intent collector, pre-loaded with file context."""
    seen: list[object] = []

    def _on_intent(intent: object) -> None:
        seen.append(intent)

    t = ImageToolbarWidget(on_intent=_on_intent)
    t.set_file_ext('f', 0, 1, channel_options=['0', '1'], roi_options=[1, 2, 3])
    return t, seen


# ---------- _format_file_label / _channel_as_str ----------


def test_format_file_label_renders_value_or_dash() -> None:
    """``_format_file_label`` should return file id or dash for None."""
    t = ImageToolbarWidget()
    assert t._format_file_label('acq') == 'acq'
    assert t._format_file_label(None) == '—'


def test_channel_as_str_handles_none_and_int() -> None:
    """``_channel_as_str`` should map int→str and None→None."""
    t = ImageToolbarWidget()
    assert t._channel_as_str(None) is None
    assert t._channel_as_str(2) == '2'


# ---------- _parse_channel_str ----------


def test_parse_channel_str_none_returns_none() -> None:
    """``_parse_channel_str(None)`` should return None."""
    t = ImageToolbarWidget()
    assert t._parse_channel_str(None) is None


def test_parse_channel_str_parses_decimal() -> None:
    """``_parse_channel_str`` should parse an int-like string."""
    t = ImageToolbarWidget()
    assert t._parse_channel_str('7') == 7


def test_parse_channel_str_raises_on_garbage() -> None:
    """``_parse_channel_str`` should raise ValueError for non-int strings."""
    t = ImageToolbarWidget()
    with pytest.raises(ValueError, match='int-compatible'):
        t._parse_channel_str('xx')


# ---------- _parse_roi_select_value ----------


def test_parse_roi_select_value_none_returns_none() -> None:
    """None input should return None."""
    assert ImageToolbarWidget._parse_roi_select_value(None) is None


def test_parse_roi_select_value_int_passes_through() -> None:
    """Plain int input should pass through unchanged."""
    assert ImageToolbarWidget._parse_roi_select_value(3) == 3


def test_parse_roi_select_value_str_parsed_as_int() -> None:
    """Decimal string should parse to int."""
    assert ImageToolbarWidget._parse_roi_select_value('5') == 5


def test_parse_roi_select_value_whole_float_parsed() -> None:
    """Whole-number float should become int."""
    assert ImageToolbarWidget._parse_roi_select_value(4.0) == 4


def test_parse_roi_select_value_rejects_bool() -> None:
    """bool inputs should be rejected explicitly (not coerced to int)."""
    with pytest.raises(ValueError, match='bool'):
        ImageToolbarWidget._parse_roi_select_value(True)


def test_parse_roi_select_value_rejects_fractional_float() -> None:
    """Fractional floats should raise ValueError."""
    with pytest.raises(ValueError, match='whole number'):
        ImageToolbarWidget._parse_roi_select_value(1.5)


def test_parse_roi_select_value_rejects_other_types() -> None:
    """Unsupported types should raise ValueError with the type name."""
    with pytest.raises(ValueError, match='unsupported'):
        ImageToolbarWidget._parse_roi_select_value([1])


# ---------- set_enabled_ext ----------


def test_set_enabled_ext_disables_toolbar() -> None:
    """``set_enabled_ext`` should update internal enabled state."""
    t = ImageToolbarWidget()
    t.set_enabled_ext(False)
    assert t._enabled is False
    t.set_enabled_ext(True)
    assert t._enabled is True


# ---------- set_channel_options_ext / set_roi_options_ext ----------


def test_set_channel_options_ext_updates_options_preserving_selection() -> None:
    """``set_channel_options_ext`` should replace options keeping current selection."""
    t = ImageToolbarWidget()
    t.set_file_ext('f', 0, 1, channel_options=['0', '1'], roi_options=[1])

    t.set_channel_options_ext(['0', '1', '2'])

    assert t.get_channel_options() == ['0', '1', '2']
    assert t.get_channel() == 0


def test_set_channel_options_ext_raises_when_selection_missing() -> None:
    """Replacing options that drop the current channel should raise."""
    t = ImageToolbarWidget()
    t.set_file_ext('f', 0, 1, channel_options=['0', '1'], roi_options=[1])

    with pytest.raises(ValueError):
        t.set_channel_options_ext(['1', '2'])


def test_set_roi_options_ext_updates_preserving_selection() -> None:
    """``set_roi_options_ext`` should replace ROI options, preserving selection."""
    t = ImageToolbarWidget()
    t.set_file_ext('f', 0, 1, channel_options=['0'], roi_options=[1, 2])

    t.set_roi_options_ext([1, 2, 3])

    assert t.get_roi_options() == [1, 2, 3]
    assert t.get_roi_id() == 1


def test_set_roi_options_ext_raises_when_selection_missing() -> None:
    """Replacing options that drop the current ROI should raise."""
    t = ImageToolbarWidget()
    t.set_file_ext('f', 0, 1, channel_options=['0'], roi_options=[1, 2])

    with pytest.raises(ValueError):
        t.set_roi_options_ext([2, 3])


# ---------- _on_channel_change ----------


def test_on_channel_change_emits_intent(hosted_toolbar) -> None:
    """``_on_channel_change`` should emit a select-channel intent."""
    t, seen = hosted_toolbar
    seen.clear()

    t._on_channel_change(SimpleNamespace(value='1'))

    assert len(seen) == 1
    assert isinstance(seen[0], ImageToolbarSelectChannelIntent)
    assert seen[0].channel == 1
    assert t.get_channel() == 1


def test_on_channel_change_respects_disabled_state(hosted_toolbar) -> None:
    """Disabled toolbar should not emit channel intents."""
    t, seen = hosted_toolbar
    t.set_enabled_ext(False)
    seen.clear()

    t._on_channel_change(SimpleNamespace(value='1'))

    assert seen == []


def test_channel_select_enabled_with_single_option() -> None:
    """Single channel should remain selectable (not disabled)."""
    t = ImageToolbarWidget(on_intent=lambda _ev: None)
    t.set_file_ext('f', 0, 1, channel_options=['0'], roi_options=[1])

    assert t.get_channel() == 0
    assert t._channel_select.enabled is True


def test_roi_select_enabled_with_single_option() -> None:
    """Single ROI should remain selectable (not disabled)."""
    t = ImageToolbarWidget(on_intent=lambda _ev: None)
    t.set_file_ext('f', 0, 5, channel_options=['0', '1'], roi_options=[5])

    assert t.get_roi_id() == 5
    assert t._roi_select.enabled is True


def test_selects_enabled_with_multiple_options(hosted_toolbar) -> None:
    """Multiple channel/ROI options keep selects enabled."""
    t, _seen = hosted_toolbar

    assert t._channel_select.enabled is True
    assert t._roi_select.enabled is True


# ---------- _on_add_click and friends ----------


def test_on_add_click_emits_add_intent(hosted_toolbar) -> None:
    """``_on_add_click`` should emit ``ImageToolbarRoiAddRequestIntent``."""
    t, seen = hosted_toolbar
    seen.clear()

    t._on_add_click()

    assert seen and isinstance(seen[-1], ImageToolbarRoiAddRequestIntent)


def test_on_add_click_noop_when_disabled(hosted_toolbar) -> None:
    """Disabled toolbar should not emit add intents."""
    t, seen = hosted_toolbar
    t.set_enabled_ext(False)
    seen.clear()

    t._on_add_click()

    assert seen == []


def test_on_delete_click_emits_intent(hosted_toolbar) -> None:
    """``_on_delete_click`` should emit a delete intent with current ROI."""
    t, seen = hosted_toolbar
    seen.clear()

    t._on_delete_click()

    assert seen and isinstance(seen[-1], ImageToolbarRoiDeleteRequestIntent)
    assert seen[-1].roi_id == 1


def test_on_delete_click_noop_when_no_current_roi() -> None:
    """Delete should not emit when no ROI is selected."""
    seen: list[object] = []
    t = ImageToolbarWidget(on_intent=lambda ev: seen.append(ev))

    t._on_delete_click()

    assert seen == []


def test_on_delete_click_noop_when_disabled(hosted_toolbar) -> None:
    """Delete should not emit when disabled."""
    t, seen = hosted_toolbar
    t.set_enabled_ext(False)
    seen.clear()

    t._on_delete_click()

    assert seen == []


# ---------- edit / OK / cancel / full-width / full-height ----------


def test_on_edit_click_enters_editing_and_emits_start(hosted_toolbar) -> None:
    """Edit click should move into EDITING mode and emit a start intent."""
    t, seen = hosted_toolbar
    seen.clear()

    t._on_edit_click()

    assert t._mode == ImageToolbarMode.EDITING
    assert seen and isinstance(seen[-1], ImageToolbarRoiEditStartIntent)
    assert seen[-1].roi_id == 1


def test_on_edit_click_noop_without_roi() -> None:
    """Edit click should not emit when no ROI is selected."""
    seen: list[object] = []
    t = ImageToolbarWidget(on_intent=lambda ev: seen.append(ev))

    t._on_edit_click()

    assert seen == []
    assert t._mode == ImageToolbarMode.IDLE


def test_on_full_width_click_emits_intent(hosted_toolbar) -> None:
    """Full-width click should emit ``ImageToolbarRoiApplyFullWidthIntent``."""
    t, seen = hosted_toolbar
    seen.clear()

    t._on_full_width_click()

    assert seen and isinstance(seen[-1], ImageToolbarRoiApplyFullWidthIntent)


def test_on_full_height_click_emits_intent(hosted_toolbar) -> None:
    """Full-height click should emit ``ImageToolbarRoiApplyFullHeightIntent``."""
    t, seen = hosted_toolbar
    seen.clear()

    t._on_full_height_click()

    assert seen and isinstance(seen[-1], ImageToolbarRoiApplyFullHeightIntent)


def test_on_ok_click_submits_and_returns_to_idle(hosted_toolbar) -> None:
    """OK in editing mode should emit submit intent and return to idle."""
    t, seen = hosted_toolbar
    t._on_edit_click()
    seen.clear()

    t._on_ok_click()

    assert any(isinstance(e, ImageToolbarRoiEditSubmitIntent) for e in seen)
    assert t._mode == ImageToolbarMode.IDLE


def test_on_cancel_click_emits_cancel_and_returns_to_idle(hosted_toolbar) -> None:
    """Cancel in editing mode should emit cancel intent and return to idle."""
    t, seen = hosted_toolbar
    t._on_edit_click()
    seen.clear()

    t._on_cancel_click()

    assert any(isinstance(e, ImageToolbarRoiEditCancelIntent) for e in seen)
    assert t._mode == ImageToolbarMode.IDLE


def test_on_full_width_noop_when_disabled(hosted_toolbar) -> None:
    """Full-width click should not emit when disabled."""
    t, seen = hosted_toolbar
    t.set_enabled_ext(False)
    seen.clear()

    t._on_full_width_click()
    t._on_full_height_click()
    t._on_ok_click()
    t._on_cancel_click()
    t._on_edit_click()

    assert seen == []


def test_emit_skipped_when_on_intent_is_none() -> None:
    """``_emit`` should be a silent no-op when no intent handler is wired."""
    t = ImageToolbarWidget(on_intent=None)
    t.set_file_ext('f', 0, 1, channel_options=['0'], roi_options=[1])

    t._on_add_click()
    t._on_delete_click()
    t._on_edit_click()


def test_editing_clicks_noop_when_current_roi_missing(hosted_toolbar) -> None:
    """Full-width / full-height / OK clicks should no-op when ROI is cleared."""
    t, seen = hosted_toolbar
    t._roi_id = None
    seen.clear()

    t._on_full_width_click()
    t._on_full_height_click()
    t._on_ok_click()

    assert seen == []
