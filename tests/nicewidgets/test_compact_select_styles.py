"""Tests for toolbar-only compact QSelect styling helpers."""

from __future__ import annotations

import nicewidgets.compact_select_styles as compact_select_styles
from nicewidgets.compact_select_styles import (
    COMPACT_SELECT_CLASS,
    COMPACT_SELECT_PROPS,
    ensure_compact_select_styles,
)
from nicewidgets.contrast_widget.contrast_widget import ContrastWidget
from nicewidgets.image_toolbar_widget.image_toolbar_widget import ImageToolbarWidget


def test_compact_select_constants() -> None:
    """Documented props and class name for toolbar selects."""
    assert COMPACT_SELECT_CLASS == 'nw-select-compact'
    assert 'dense' in COMPACT_SELECT_PROPS
    assert 'hide-bottom-space' in COMPACT_SELECT_PROPS
    assert 'options-dense' in COMPACT_SELECT_PROPS
    assert 'standout' in COMPACT_SELECT_PROPS


def test_ensure_compact_select_styles_is_idempotent(monkeypatch) -> None:
    """Second call must not register duplicate head HTML."""
    compact_select_styles._INJECTED = False
    calls: list[str] = []

    def _fake_add_head_html(html: str, *, shared: bool = False) -> None:
        calls.append(html)
        assert shared is True

    monkeypatch.setattr(compact_select_styles.ui, 'add_head_html', _fake_add_head_html)
    ensure_compact_select_styles()
    ensure_compact_select_styles()
    assert len(calls) == 1
    assert COMPACT_SELECT_CLASS in calls[0]
    compact_select_styles._INJECTED = False


def test_image_toolbar_selects_use_compact_class() -> None:
    """Channel and ROI selects apply compact class and props only on those controls."""
    toolbar = ImageToolbarWidget()
    for select in (toolbar._channel_select, toolbar._roi_select):
        assert COMPACT_SELECT_CLASS in select.classes
        assert 'hide-bottom-space' in str(select._props)  # noqa: SLF001


def test_contrast_lut_select_uses_compact_class() -> None:
    """Color LUT select applies compact class and props."""
    widget = ContrastWidget()
    assert COMPACT_SELECT_CLASS in widget._lut_select.classes
    assert 'hide-bottom-space' in str(widget._lut_select._props)  # noqa: SLF001
