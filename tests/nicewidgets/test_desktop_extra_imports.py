"""Validate optional desktop-extra packages when installed."""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.desktop_extra

_DESKTOP_MODULES = ('pyperclip', 'pyperclipimg', 'webview')


def _desktop_extra_installed() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in _DESKTOP_MODULES)


@pytest.mark.skipif(
    not _desktop_extra_installed(),
    reason='optional desktop extra not installed (pyperclip, pyperclipimg, pywebview)',
)
def test_desktop_extra_imports() -> None:
    """Desktop profile must provide native clipboard and pywebview imports."""
    import pyperclip
    import pyperclipimg
    import webview

    assert pyperclip is not None
    assert pyperclipimg is not None
    assert webview is not None
