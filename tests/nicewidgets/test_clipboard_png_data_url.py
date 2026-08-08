"""Tests for PNG data-URL helpers in :mod:`nicewidgets.utils.clipboard`."""

from __future__ import annotations

import base64

import pytest

from nicewidgets.utils.clipboard import png_bytes_from_data_url

# Minimal 1x1 PNG.
_PNG_BYTES = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQ'
    'AAAABJRU5ErkJggg=='
)


def test_png_bytes_from_data_url_round_trip() -> None:
    """A valid PNG data URL should decode to the original bytes."""
    data_url = 'data:image/png;base64,' + base64.b64encode(_PNG_BYTES).decode('ascii')
    assert png_bytes_from_data_url(data_url) == _PNG_BYTES


def test_png_bytes_from_data_url_rejects_non_png() -> None:
    """Non-PNG data URLs should fail fast."""
    with pytest.raises(ValueError, match='data:image/png;base64'):
        png_bytes_from_data_url('data:image/jpeg;base64,AAAA')
