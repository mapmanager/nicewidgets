"""Tests for desktop pywebview detection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nicewidgets.utils import clipboard as clipboard_mod
from nicewidgets.utils import desktop as desktop_mod


def test_is_pywebview_desktop_true_for_nicegui_native_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy ``ui.run(native=True)`` should report desktop mode."""
    monkeypatch.setattr(
        desktop_mod.app,
        'native',
        SimpleNamespace(main_window=object()),
        raising=False,
    )

    assert desktop_mod.is_pywebview_desktop() is True


def test_is_pywebview_desktop_true_for_option_c_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Option C manual ``webview.create_window`` should report desktop mode."""
    monkeypatch.setattr(desktop_mod.app, 'native', SimpleNamespace(main_window=None), raising=False)

    fake_webview = SimpleNamespace(windows=[object()])
    monkeypatch.setitem(__import__('sys').modules, 'webview', fake_webview)

    assert desktop_mod.is_pywebview_desktop() is True


def test_is_pywebview_desktop_false_for_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Browser sessions without pywebview windows should report False."""
    monkeypatch.setattr(desktop_mod.app, 'native', SimpleNamespace(main_window=None), raising=False)

    fake_webview = SimpleNamespace(windows=[])
    monkeypatch.setitem(__import__('sys').modules, 'webview', fake_webview)

    assert desktop_mod.is_pywebview_desktop() is False


def test_clipboard_module_reexports_is_pywebview_desktop() -> None:
    """``clipboard`` keeps a stable import path for desktop detection."""
    assert clipboard_mod.is_pywebview_desktop is desktop_mod.is_pywebview_desktop
