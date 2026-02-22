"""Smoke tests for LazySection and LazySectionConfig."""

from __future__ import annotations

from typing import Any, Callable
from unittest.mock import MagicMock

import pytest

import nicewidgets.utils.lazy_section as ls_mod
from nicewidgets.utils.lazy_section import LazySection, LazySectionConfig


def test_lazy_section_config_import_and_instantiate() -> None:
    """LazySectionConfig can be imported and instantiated with defaults."""
    cfg = LazySectionConfig()
    assert cfg.render_once is True
    assert cfg.clear_on_close is False
    assert cfg.show_spinner is True
    assert cfg.on_clear is None


def test_lazy_section_config_options() -> None:
    """LazySectionConfig accepts custom options."""
    cb = lambda: None
    cfg = LazySectionConfig(
        render_once=False,
        clear_on_close=True,
        show_spinner=False,
        on_clear=cb,
    )
    assert cfg.render_once is False
    assert cfg.clear_on_close is True
    assert cfg.show_spinner is False
    assert cfg.on_clear is cb


@pytest.mark.requires_nicegui
def test_lazy_section_import_and_instantiate(monkeypatch: pytest.MonkeyPatch) -> None:
    """LazySection can be imported and instantiated with mocked UI."""
    fake_expansion = MagicMock()
    fake_expansion.on = MagicMock()
    fake_expansion.classes = MagicMock(return_value=fake_expansion)
    fake_expansion.__enter__ = MagicMock(return_value=fake_expansion)
    fake_expansion.__exit__ = MagicMock(return_value=None)

    fake_column = MagicMock()
    fake_container = MagicMock()
    fake_column.classes = MagicMock(return_value=fake_column)
    fake_column.__enter__ = MagicMock(return_value=fake_container)
    fake_column.__exit__ = MagicMock(return_value=None)

    fake_label = MagicMock()
    fake_label.classes = MagicMock(return_value=fake_label)

    fake_spinner = MagicMock()
    fake_spinner.visible = False
    fake_spinner.classes = MagicMock(return_value=fake_spinner)

    def fake_expansion_factory(*args: Any, **kwargs: Any) -> Any:
        return fake_expansion

    def fake_column_factory(*args: Any, **kwargs: Any) -> Any:
        return fake_column

    def fake_label_factory(text: str = "") -> Any:
        return fake_label

    def fake_spinner_factory(*args: Any, **kwargs: Any) -> Any:
        return fake_spinner

    fake_ui = MagicMock()
    fake_ui.expansion = fake_expansion_factory
    fake_ui.column = fake_column_factory
    fake_ui.label = fake_label_factory
    fake_ui.spinner = fake_spinner_factory

    monkeypatch.setattr(ls_mod, "ui", fake_ui, raising=True)

    def _render_fn(container: Any) -> None:
        pass

    section = LazySection(
        "Test",
        render_fn=_render_fn,
        config=LazySectionConfig(render_once=True, clear_on_close=False),
    )

    assert section._title == "Test"
    assert section._cfg.render_once is True
    assert section._cfg.clear_on_close is False
    fake_expansion.on.assert_called_once_with("update:model-value", section._on_model_value)
