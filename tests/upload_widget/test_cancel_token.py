# tests/upload_widget/test_cancel_token.py
from __future__ import annotations

from nicewidgets.upload_widget.upload_widget import CancelToken


def test_cancel_token_sets_flag() -> None:
    t = CancelToken()
    assert t.cancelled is False
    t.cancel()
    assert t.cancelled is True
