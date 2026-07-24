"""Tests for :class:`nicewidgets.smart_expansion_widget.smart_expansion.SmartExpansion`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nicewidgets.smart_expansion_widget.smart_expansion import SmartExpansion


@pytest.fixture
def callbacks() -> dict[str, list[str]]:
    return {'open': [], 'close': []}


@pytest.fixture
def expansion(callbacks: dict[str, list[str]]) -> SmartExpansion:
    return SmartExpansion(
        'Section',
        initially_open=False,
        on_open=lambda: callbacks['open'].append('open'),
        on_close=lambda: callbacks['close'].append('close'),
    )


def test_initially_closed_apply_initial_state_fires_on_close(expansion: SmartExpansion, callbacks: dict[str, list[str]]) -> None:
    """Closed expansions should invoke ``on_close`` once during initial apply."""
    expansion.apply_initial_state()

    assert expansion.is_open is False
    assert callbacks['close'] == ['close']
    assert callbacks['open'] == []


def test_initially_open_apply_initial_state_fires_on_open(callbacks: dict[str, list[str]]) -> None:
    """Open expansions should invoke ``on_open`` once during initial apply."""
    expansion = SmartExpansion(
        'Section',
        initially_open=True,
        on_open=lambda: callbacks['open'].append('open'),
        on_close=callbacks['close'].append,
    )

    expansion.apply_initial_state()

    assert expansion.is_open is True
    assert callbacks['open'] == ['open']
    assert callbacks['close'] == []


def test_value_change_open_and_close(expansion: SmartExpansion, callbacks: dict[str, list[str]]) -> None:
    """User-driven value changes should fire open then close callbacks."""
    expansion.apply_initial_state()
    callbacks['open'].clear()
    callbacks['close'].clear()

    expansion._on_value_change(SimpleNamespace(value=True))
    expansion._on_value_change(SimpleNamespace(value=False))

    assert callbacks['open'] == ['open']
    assert callbacks['close'] == ['close']


def test_programmatic_open_and_close(expansion: SmartExpansion, callbacks: dict[str, list[str]]) -> None:
    """Programmatic ``open`` / ``close`` should dispatch lifecycle callbacks."""
    expansion.apply_initial_state()
    callbacks['open'].clear()
    callbacks['close'].clear()

    expansion.open()
    expansion.close()

    assert expansion.is_open is False
    assert callbacks['open'] == ['open']
    assert callbacks['close'] == ['close']


def test_idempotent_close_does_not_double_fire(expansion: SmartExpansion, callbacks: dict[str, list[str]]) -> None:
    """Repeated close requests should not invoke ``on_close`` twice."""
    expansion.apply_initial_state()
    callbacks['close'].clear()

    expansion._on_value_change(SimpleNamespace(value=False))
    expansion._on_value_change(SimpleNamespace(value=False))

    assert callbacks['close'] == []


def test_apply_initial_state_is_idempotent(callbacks: dict[str, list[str]]) -> None:
    """``apply_initial_state`` should only dispatch callbacks once."""
    expansion = SmartExpansion(
        'Section',
        initially_open=True,
        on_open=lambda: callbacks['open'].append('open'),
        on_close=lambda: callbacks['close'].append('close'),
    )

    expansion.apply_initial_state()
    expansion.apply_initial_state()

    assert callbacks['open'] == ['open']
    assert callbacks['close'] == []


def test_expansion_property_returns_inner_element(expansion: SmartExpansion) -> None:
    """``expansion`` should expose the wrapped NiceGUI element."""
    assert expansion.expansion is expansion._expansion
