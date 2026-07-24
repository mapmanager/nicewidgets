"""Smoke tests for ``ImageToolbarWidget`` (NiceGUI element construction)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nicewidgets.image_toolbar_widget.image_toolbar_widget import ImageToolbarWidget
from nicewidgets.image_toolbar_widget.intent import ImageToolbarSelectRoiIntent


@pytest.fixture
def toolbar() -> ImageToolbarWidget:
    return ImageToolbarWidget()


def test_set_file_ext_updates_internal_state(toolbar: ImageToolbarWidget) -> None:
    toolbar.set_file_ext('acq-1', 0, 2, channel_options=['0', '1'], roi_options=[1, 2, 3])
    assert toolbar.get_file_id() == 'acq-1'
    assert toolbar.get_channel() == 0
    assert toolbar.get_roi_id() == 2
    assert toolbar.get_channel_options() == ['0', '1']
    assert toolbar.get_roi_options() == [1, 2, 3]


def test_set_roi_options_and_selection_ext(toolbar: ImageToolbarWidget) -> None:
    toolbar.set_file_ext('f', 0, 1, channel_options=['0'], roi_options=[1, 2])
    toolbar.set_roi_options_and_selection_ext([2, 3], 3)
    assert toolbar.get_roi_options() == [2, 3]
    assert toolbar.get_roi_id() == 3


def test_set_file_ext_rejects_roi_none_when_options_nonempty(toolbar: ImageToolbarWidget) -> None:
    with pytest.raises(ValueError, match='roi_id is required'):
        toolbar.set_file_ext('f', 0, None, channel_options=['0'], roi_options=[1])


def test_programmatic_updates_do_not_emit_intents(toolbar: ImageToolbarWidget) -> None:
    seen: list[object] = []

    def _on_intent(ev: object) -> None:
        seen.append(ev)

    t2 = ImageToolbarWidget(on_intent=_on_intent)
    t2.set_file_ext('f', 0, 1, channel_options=['0'], roi_options=[1])
    t2.set_channel_ext(0)
    t2.set_roi_ext(1)
    t2.set_roi_options_and_selection_ext([1, 2], 2)
    assert seen == []


def test_on_intent_receives_roi_select_when_user_changes(toolbar: ImageToolbarWidget) -> None:
    """Simulate ROI select handler path (no browser)."""
    seen: list[object] = []

    def _on_intent(ev: object) -> None:
        seen.append(ev)

    t = ImageToolbarWidget(on_intent=_on_intent)
    t.set_file_ext('f', 0, 1, channel_options=['0'], roi_options=[1, 2])
    t._on_roi_change(SimpleNamespace(value=2))
    assert len(seen) == 1
    assert isinstance(seen[0], ImageToolbarSelectRoiIntent)
    assert seen[0].roi_id == 2
